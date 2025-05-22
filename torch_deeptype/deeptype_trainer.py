import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Optional, Sequence, Tuple

from .deeptype_model import DeeptypeModel
from .losses import SparsityLoss, ClusterRepresentationLoss

class DeeptypeTrainer:
    def __init__(
        self,
        model: DeeptypeModel,
        train_loader: DataLoader,
        primary_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_clusters: int,
        *,
        sparsity_weight: float = 0.006,
        cluster_weight: float = 1.2,
        verbose: bool = False,
        callback_supervised: Optional[Callable[[int, DeeptypeModel, float, float], None]] = None,
        callback_supervised_unsupervised: Optional[Callable[[int, DeeptypeModel, float, float, float], None]] = None
    ):
        """
        Args:
          model: a DeeptypeModel instance
          train_loader: yields (inputs, targets) for stage 1
          primary_loss_fn: e.g. nn.MSELoss() or nn.CrossEntropyLoss()
          clusters: Number of clusters to create
          sparsity_weight: Weight for the sparsity loss
          cluster_weight: Weight for the cluster representation loss
          device: torch.device; defaults to CUDA if available
          verbose: whether to print per‐epoch stats
        """
        self.model            = model
        self.train_loader     = train_loader
        self.primary_loss_fn  = primary_loss_fn
        self.sparsity_weight  = sparsity_weight
        self.cluster_weight   = cluster_weight
        self.verbose          = verbose
        self.num_clusters     = num_clusters

        # instantiate the two regularizers once
        self._sparsity_loss = SparsityLoss()
        self._cluster_loss  = ClusterRepresentationLoss()
        self.callback_supervised = callback_supervised
        self.callback_supervised_unsupervised = callback_supervised_unsupervised
        self._should_stop = False
        
    def compute_losses(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        do_cluster: bool = False,
        centers: Optional[Sequence[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (primary_loss, sparsity_loss, cluster_loss).
        If do_cluster is False, cluster_loss is None.
        """
        outputs = self.model(inputs)
        lp = self.primary_loss_fn(outputs, targets)
        ls = self._sparsity_loss(self.model)
        lc = None
        if do_cluster and centers is not None:
            lc = self._cluster_loss(self.model, inputs, centers)
        return lp, ls, lc

    def train_supervised(self, num_epochs: int, lr: float) -> None:
        """
        Phase 1: train on primary loss + α * sparsity.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        n_samples = len(self.train_loader.dataset)

        for epoch in range(1, num_epochs + 1):
            if self._should_stop:
                print("Ending phase 1 (training stopped)")
                return
            total_primary  = 0.0
            total_sparsity = 0.0

            for inputs, targets in self.train_loader:
                optimizer.zero_grad()

                lp, ls, _ = self.compute_losses(inputs, targets, do_cluster=False)
                loss = lp + self.sparsity_weight * ls

                loss.backward()
                optimizer.step()

                total_primary  += lp.item() * inputs.size(0)
                total_sparsity += ls.item() * inputs.size(0)

            avg_p = total_primary  / n_samples
            avg_s = total_sparsity / n_samples
            
            if self.verbose:
                print(
                    f"[Epoch {epoch}/{num_epochs}]  "
                    f"Primary: {avg_p:0.4f}  "
                    f"Sparsity: {avg_s:0.4f}"
                )
                
            if self.callback_supervised:
                self.callback_supervised(epoch, self.model, avg_p, avg_s)

        if self.verbose:
            print("→ Phase 1 training complete.\n")
            
    def _get_cluster_centers(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Run KMeans on the model's hidden representations of `inputs`
        to find `self.num_clusters` clusters, then for each input
        return the cluster-center tensor it belongs to.
        """
        # 1) Get hidden reps
        self.model.eval()
        with torch.no_grad():
            reps: torch.Tensor = self.model.get_hidden_representations(inputs.to(next(self.model.parameters()).device))

        # 2) Move to CPU numpy for sklearn
        reps_np = reps.detach().cpu().numpy()

        # 3) Fit KMeans
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init="auto")
        kmeans.fit(reps_np)

        # 4) Grab centers and labels
        centers_np = kmeans.cluster_centers_
        labels     = kmeans.labels_

        # 5) Build list of Tensors, one center per input
        device = reps.device
        dtype  = reps.dtype
        cluster_centers = [
            torch.from_numpy(centers_np[label]).to(device).type(dtype)
            for label in labels
        ]
        return cluster_centers
            
    def train_supervised_unsupervised(
        self,
        num_epochs: int,
        lr: float,
        train_steps_per_batch: int = 10
    ) -> None:
        """
        Phase 2: train on primary + α * sparsity + β * cluster‐rep loss.
        Runs `train_steps_per_batch` inner updates per batch.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        n_samples = len(self.train_loader.dataset)

        for epoch in range(1, num_epochs + 1):
            if self._should_stop:
                print("Ending phase 2 (training stopped)")
                return
            total_primary    = 0.0
            total_sparsity   = 0.0
            total_clustering = 0.0

            for inputs, targets in self.train_loader:
                centers = self._get_cluster_centers(inputs)

                for _ in range(train_steps_per_batch):
                    optimizer.zero_grad()
                    lp, ls, lc = self.compute_losses(
                        inputs, targets,
                        do_cluster=True,
                        centers=centers
                    )
                    loss = lp + self.sparsity_weight * ls + self.cluster_weight * lc

                    loss.backward()
                    optimizer.step()

                    total_primary    += lp.item() * inputs.size(0)
                    total_sparsity   += ls.item() * inputs.size(0)
                    total_clustering += lc.item() * inputs.size(0)

            denom = n_samples * train_steps_per_batch
            avg_p = total_primary    / denom
            avg_s = total_sparsity   / denom
            avg_c = total_clustering / denom

            if self.verbose:
                print(
                    f"[Phase 2 | Epoch {epoch}/{num_epochs}]  "
                    f"Primary: {avg_p:0.4f}  "
                    f"Sparsity: {avg_s:0.4f}  "
                    f"Cluster: {avg_c:0.4f}"
                )
                
            if self.callback_supervised_unsupervised:
                self.callback_supervised_unsupervised(epoch, self.model, avg_p, avg_s, avg_c)

        if self.verbose:
            print("→ Phase 2 training complete.\n")
            
    def train(
        self,
        main_epochs: int,
        main_lr: float,
        pretrain_epochs: int = 10,
        pretrain_lr: Optional[float] = None,
        train_steps_per_batch: int = 10
    ) -> None:
        """
        Run a two‐stage training schedule:
        
        1. Pretrain for `pretrain_epochs` using only the primary + sparsity loss.
        2. Then train for `main_epochs` using primary + sparsity + clustering loss
           with `train_steps_per_batch` inner updates per batch.

        Args:
            main_epochs (int):
                Number of epochs for the joint supervised+unsupervised phase.
            main_lr (float):
                Learning rate for the joint phase.
            pretrain_epochs (int, optional):
                Number of epochs for the initial supervised+sparsity-only phase.
                Defaults to 10.
            pretrain_lr (float, optional):
                Learning rate for the pretrain phase. If `None`, uses `main_lr`.
                Defaults to None.
            train_steps_per_batch (int, optional):
                How many optimization steps to take on each batch during the
                joint phase. More steps can help the cluster loss converge
                within each batch. Defaults to 10.

        Returns:
            None: the model is updated in-place.
        """
        self._should_stop = False
        # 1) Pretrain (phase 1)
        lr1 = pretrain_lr if pretrain_lr is not None else main_lr
        if self.verbose:
            print(f"Starting Phase 1: {pretrain_epochs} epochs @ lr={lr1}")
        self.train_supervised(num_epochs=pretrain_epochs, lr=lr1)

        # 2) Joint supervised + unsupervised (phase 2)
        if self.verbose:
            print(f"Starting Phase 2: {main_epochs} epochs @ lr={main_lr} "
                  f"with {train_steps_per_batch} steps/batch")
        self.train_supervised_unsupervised(
            num_epochs=main_epochs,
            lr=main_lr,
            train_steps_per_batch=train_steps_per_batch
        )

        if self.verbose:
            print("All training complete.")
            
    def get_clusters(
        self,
        dataset: Dataset
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Cluster the entire dataset's hidden representations in one go.

        Args:
            dataset (torch.utils.data.Dataset):
                Should return either:
                  - (input, target) pairs, or
                  - input-only
                for each __getitem__.

        Returns:
            cluster_centers (Tensor[num_clusters, hidden_dim]):
                The KMeans centroids with the same dtype
                as your model's representations.
            labels (np.ndarray[N]):
                For each example in dataset order, the assigned cluster index
                (0 … num_clusters-1).
        """
        # 1) Pull all inputs into one big tensor
        inputs_list = []
        for item in dataset:
            x = item[0] if isinstance(item, (list, tuple)) else item
            # ensure a batch dimension
            inputs_list.append(x.unsqueeze(0))
        inputs_all = torch.cat(inputs_list, dim=0)

        # 2) Get hidden reps
        self.model.eval()
        with torch.no_grad():
            reps = self.model.get_hidden_representations(inputs_all)

        # 3) KMeans on CPU numpy
        reps_np = reps.cpu().numpy()
        kmeans  = KMeans(n_clusters=self.num_clusters, random_state=0, n_init="auto")
        kmeans.fit(reps_np)

        # 4) Convert back to torch
        centers_np = kmeans.cluster_centers_           # (num_clusters, hidden_dim)
        labels     = kmeans.labels_                    # (N,)

        cluster_centers = (
            torch.from_numpy(centers_np)
                 .type(reps.dtype)
        )
        return cluster_centers, labels
    
    def stop(self):
        """
        Stop the trainer in the middle of training
        
        NOTE: clear_stop() must be called for training to continue (e.g. if stopping the supervised phase early and
        then proceeding to the unsupervised phase.)
        """
        self._should_stop = True
        
    def clear_stop(self):
        """
        Clear the stop flag. Required to continue training after stop() is called.
        """
        self._should_stop = False

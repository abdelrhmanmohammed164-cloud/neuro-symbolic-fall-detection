"""
graph.py
14-joint skeleton graph with ST-GCN spatial-configuration partitioning.

Partitions (Yan et al., 2018):
    A[0]  self-links
    A[1]  centripetal: neighbour is closer to the centre joint (the Neck)
    A[2]  centrifugal: neighbour is farther from the centre joint
Each partition is column-normalised (A @ D^-1), as in the original ST-GCN.

Edit JOINTS / EDGES / FLIP_PAIRS here if your keypoint ordering differs.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1. Joint definition  (index -> name). Verify this against your .npy files.
# ---------------------------------------------------------------------------
JOINTS = [
    "Head",        # 0  (RTMO/COCO nose)
    "Neck",        # 1  (midpoint of the two shoulders, computed in preprocessing)
    "R_Shoulder",  # 2
    "R_Elbow",     # 3
    "R_Wrist",     # 4
    "L_Shoulder",  # 5
    "L_Elbow",     # 6
    "L_Wrist",     # 7
    "R_Hip",       # 8
    "R_Knee",      # 9
    "R_Ankle",     # 10
    "L_Hip",       # 11
    "L_Knee",      # 12
    "L_Ankle",     # 13
]
NUM_JOINTS = len(JOINTS)
J = {name: i for i, name in enumerate(JOINTS)}

CENTER = J["Neck"]

# Bones (undirected). The tree is rooted at the Neck.
EDGES = [
    (J["Head"], J["Neck"]),
    (J["Neck"], J["R_Shoulder"]), (J["R_Shoulder"], J["R_Elbow"]), (J["R_Elbow"], J["R_Wrist"]),
    (J["Neck"], J["L_Shoulder"]), (J["L_Shoulder"], J["L_Elbow"]), (J["L_Elbow"], J["L_Wrist"]),
    (J["Neck"], J["R_Hip"]), (J["R_Hip"], J["R_Knee"]), (J["R_Knee"], J["R_Ankle"]),
    (J["Neck"], J["L_Hip"]), (J["L_Hip"], J["L_Knee"]), (J["L_Knee"], J["L_Ankle"]),
]

# Left/right pairs, used by horizontal-flip augmentation.
FLIP_PAIRS = [
    (J["R_Shoulder"], J["L_Shoulder"]), (J["R_Elbow"], J["L_Elbow"]), (J["R_Wrist"], J["L_Wrist"]),
    (J["R_Hip"], J["L_Hip"]), (J["R_Knee"], J["L_Knee"]), (J["R_Ankle"], J["L_Ankle"]),
]

# COCO-17 (RTMO output) index for each joint above. "Neck" is not a COCO joint:
# it is built from the two shoulders (see coco17_to_urfd14).
COCO17_INDEX = {
    "Head": 0, "R_Shoulder": 6, "R_Elbow": 8, "R_Wrist": 10,
    "L_Shoulder": 5, "L_Elbow": 7, "L_Wrist": 9,
    "R_Hip": 12, "R_Knee": 14, "R_Ankle": 16,
    "L_Hip": 11, "L_Knee": 13, "L_Ankle": 15,
}


def coco17_to_urfd14(kpts: np.ndarray) -> np.ndarray:
    """Convert RTMO/COCO-17 keypoints (..., 17, 3) [x, y, conf] to this 14-joint layout.

    Neck = midpoint of the shoulders; its confidence = min of the two shoulder confidences.
    """
    out = np.zeros(kpts.shape[:-2] + (NUM_JOINTS, kpts.shape[-1]), dtype=kpts.dtype)
    for name, ci in COCO17_INDEX.items():
        out[..., J[name], :] = kpts[..., ci, :]
    ls, rs = kpts[..., 5, :], kpts[..., 6, :]
    out[..., CENTER, :2] = (ls[..., :2] + rs[..., :2]) / 2.0
    if kpts.shape[-1] > 2:
        out[..., CENTER, 2] = np.minimum(ls[..., 2], rs[..., 2])
    return out


class Graph:
    """Adjacency tensor A of shape (3, V, V) for the 14-joint skeleton."""

    def __init__(self, max_hop: int = 1):
        self.num_node = NUM_JOINTS
        self.edges = EDGES
        self.center = CENTER
        self.max_hop = max_hop
        self.hop_dis = self._hop_distance()
        self.A = self._spatial_partition()

    def _hop_distance(self) -> np.ndarray:
        V = self.num_node
        adj = np.zeros((V, V))
        for i, j in self.edges:
            adj[i, j] = adj[j, i] = 1
        hop = np.full((V, V), np.inf)
        reach = [np.linalg.matrix_power(adj + np.eye(V), d) > 0 for d in range(self.max_hop + 1)]
        for d in range(self.max_hop, -1, -1):
            hop[reach[d]] = d
        # Full shortest-path distance to the centre (needed for partitioning).
        self.dist_to_center = self._bfs(adj, self.center)
        return hop

    @staticmethod
    def _bfs(adj: np.ndarray, src: int) -> np.ndarray:
        dist = np.full(adj.shape[0], np.inf)
        dist[src], frontier = 0, [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in np.nonzero(adj[u])[0]:
                    if dist[v] == np.inf:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        return dist

    @staticmethod
    def _normalize(a: np.ndarray) -> np.ndarray:
        deg = a.sum(0)
        dinv = np.zeros_like(deg)
        dinv[deg > 0] = 1.0 / deg[deg > 0]
        return a @ np.diag(dinv)

    def _spatial_partition(self) -> np.ndarray:
        V = self.num_node
        adj = (self.hop_dis <= self.max_hop).astype(float)
        norm = self._normalize(adj)
        a_self, a_in, a_out = np.zeros((V, V)), np.zeros((V, V)), np.zeros((V, V))
        d = self.dist_to_center
        for i in range(V):
            for j in range(V):
                if self.hop_dis[j, i] > self.max_hop:
                    continue
                if d[j] == d[i]:
                    a_self[j, i] = norm[j, i]
                elif d[j] < d[i]:      # j is nearer the centre than i -> centripetal
                    a_in[j, i] = norm[j, i]
                else:                  # j is farther from the centre -> centrifugal
                    a_out[j, i] = norm[j, i]
        return np.stack([a_self, a_in, a_out]).astype(np.float32)


if __name__ == "__main__":
    g = Graph()
    print("A shape:", g.A.shape)
    print("distance to Neck:", dict(zip(JOINTS, g.dist_to_center.astype(int))))
    for k, name in enumerate(["self", "centripetal", "centrifugal"]):
        print(f"{name:12s} non-zero entries: {int((g.A[k] > 0).sum())}")

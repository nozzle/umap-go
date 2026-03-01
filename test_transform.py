import numpy as np
import umap
import sys

np.random.seed(42)
X_train = np.random.rand(100, 4).astype(np.float32)
X_test = np.random.rand(20, 4).astype(np.float32)

model = umap.UMAP(n_neighbors=5, min_dist=0.1, n_epochs=100, random_state=42)
embedding_train = model.fit_transform(X_train)
embedding_test = model.transform(X_test)

np.savetxt("transform_test.csv", embedding_test, delimiter=",")

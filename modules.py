import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.optimize import leastsq
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy import optimize
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot_paths(path_XYs, colors=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(path_XYs):
        c = colors[i % len(colors)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    return fig

def extract_features(path_XYs):
    features = []
    for paths in path_XYs:
        for path in paths:
            distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
            angles = np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0]))
            features.append([np.mean(distances), np.std(distances), np.mean(angles), np.std(angles)])
    features = np.array(features)
    return features

def regularize_curves(path_XYs, n_clusters=3):
    features = extract_features(path_XYs)
    if features.shape[0] < n_clusters:
        raise ValueError(f"Number of samples ({features.shape[0]}) should be >= number of clusters ({n_clusters})")
    kmeans = KMeans(n_clusters=n_clusters).fit(features)
    labels = kmeans.labels_
    return labels

def classify_shape(XY):
    if is_line(XY):
        return 'line'
    elif is_circle(XY):
        return 'circle'
    elif is_rectangle(XY):
        return 'rectangle'
    elif is_ellipse(XY):
        return 'ellipse'
    else:
        return 'other'

def is_line(XY, threshold=0.99):
    if len(XY) < 3:
        return True
    _, singular_values, _ = np.linalg.svd(XY - np.mean(XY, axis=0))
    return singular_values[0] / np.sum(singular_values) > threshold

def is_circle(XY, tolerance=0.1):
    if len(XY) < 5:
        return False
    center = np.mean(XY, axis=0)
    radius = np.mean(np.linalg.norm(XY - center, axis=1))
    max_deviation = np.max(np.abs(np.linalg.norm(XY - center, axis=1) - radius))
    return max_deviation / radius < tolerance

def is_rectangle(XY, angle_tolerance=5, side_tolerance=0.1):
    if len(XY) < 4:
        return False
    hull = ConvexHull(XY)
    if len(hull.vertices) != 4:
        return False
    corners = XY[hull.vertices]
    angles = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(np.diff(corners[:, 1], append=corners[0, 1]),
                                                            np.diff(corners[:, 0], append=corners[0, 0]))))))
    if not np.all(np.abs(angles - 90) < angle_tolerance):
        return False
    sides = np.linalg.norm(np.diff(corners, axis=0, append=[corners[0]]), axis=1)
    return np.std(sides) / np.mean(sides) < side_tolerance

def is_ellipse(XY, tolerance=0.1):
    if len(XY) < 5:
        return False
    center = np.mean(XY, axis=0)
    centered = XY - center
    U, s, Vt = np.linalg.svd(centered)
    a, b = s[:2]
    predicted = center + np.dot(U[:, :2] * s[:2], Vt[:2, :])
    error = np.linalg.norm(XY - predicted, axis=1)
    return np.max(error) / np.mean([a, b]) < tolerance

def fit_ellipse(XY):
    x = XY[:, 0]
    y = XY[:, 1]
    D = np.column_stack((x**2, x*y, y**2, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a

def build_symmetry_detection_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(2,)), 
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_symmetry_detection_model(model, curves, labels, epochs=10, batch_size=32):
    model.fit(curves, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

def complete_curve(points):
    if len(points) < 3:
        return complete_line(points)
    
    shape_type = classify_shape(points)
    
    if shape_type == 'circle':
        return complete_circle(points)
    elif shape_type == 'ellipse':
        return complete_ellipse(points)
    elif shape_type == 'line':
        return complete_line(points)
    else:
        return complete_generic_curve(points)

def complete_circle(points):
    center = np.mean(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    radius = np.mean(radii)
    
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack((x, y))

def complete_ellipse(points):
    center = np.mean(points, axis=0)
    cov = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    a = np.sqrt(eigenvalues[0]) * 2
    b = np.sqrt(eigenvalues[1]) * 2
    
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
    y = center[1] + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
    return np.column_stack((x, y))

def complete_line(points):
    x = points[:, 0]
    y = points[:, 1]
    
    if len(points) == 2:
        t = np.linspace(0, 1, 100)
        x_interp = np.interp(t, [0, 1], x)
        y_interp = np.interp(t, [0, 1], y)
    else:
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        x_range = np.linspace(np.min(x), np.max(x), 100)
        y_interp = model.predict(x_range.reshape(-1, 1))
        x_interp = x_range
    
    return np.column_stack((x_interp, y_interp))

def complete_generic_curve(points):
    t = np.linspace(0, 1, len(points))
    t_interp = np.linspace(0, 1, 100)
    
    x_interp = np.interp(t_interp, t, points[:, 0])
    y_interp = np.interp(t_interp, t, points[:, 1])
    
    return np.column_stack((x_interp, y_interp))

def complete_curves(path_XYs):
    completed_curves = []
    for path in path_XYs:
        completed_path = []
        for curve in path:
            completed_curve = complete_curve(curve)
            completed_path.append(completed_curve)
        completed_curves.append(completed_path)
    return completed_curves

def plot_symmetry_detection(path_XYs, symmetry_model):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    for path in path_XYs:
        for curve in path:
            ax1.plot(curve[:, 0], curve[:, 1])
    ax1.set_title("Original Curves")
    ax1.set_aspect('equal')

    for path in path_XYs:
        for curve in path:
            symmetry_scores = symmetry_model.predict(curve)
            colors = plt.cm.RdYlGn(symmetry_scores.flatten())
            ax2.scatter(curve[:, 0], curve[:, 1], c=colors, s=5)
            
            if np.mean(symmetry_scores) > 0.5: 
                pca = PCA(n_components=2)
                pca.fit(curve)
                
                v = pca.components_[0]
                centroid = np.mean(curve, axis=0)
                t = np.linspace(-1, 1, 100)
                line_points = centroid + np.outer(t, v)
                
                ax2.plot(line_points[:, 0], line_points[:, 1], 'r--', linewidth=2)

    ax2.set_title("Symmetry Detection Results")
    ax2.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2)
    cbar.set_label('Symmetry Score')

    return fig

import numpy as np
from scipy.spatial.transform import Rotation as R
import ProjectiveGeometry23.pluecker as pluecker
from ProjectiveGeometry23.utils import homogenize, dehomogenize
from xray_epipolar_consistency.parameterization.gantry_angle import rotation

def test_rotation_z_axis():
    # Z-axis Plücker line: joins (0,0,0,1) and (0,0,1,1)
    A = np.array([0.0, 0.0, 0.0, 1.0]).reshape(-1, 1)
    B = np.array([0.0, 0.0, 1.0, 1.0]).reshape(-1, 1)
    L_z = pluecker.join_points(A, B)
    
    # 90 degrees rotation
    angle = np.pi / 2.0
    T = rotation(L_z, angle)
    
    # Rotate point on X-axis (1, 0, 0, 1)
    X = np.array([1.0, 0.0, 0.0, 1.0]).reshape(-1, 1)
    X_rot = T @ X
    X_rot = dehomogenize(X_rot)
    
    # Expected: (0, 1, 0)
    np.testing.assert_allclose(X_rot.flatten(), [0.0, 1.0, 0.0], atol=1e-7)
    
    # Rotate point on the rotation axis itself (0, 0, 2, 1)
    Y = np.array([0.0, 0.0, 2.0, 1.0]).reshape(-1, 1)
    Y_rot = T @ Y
    Y_rot = dehomogenize(Y_rot)
    
    # Expected: (0, 0, 2)
    np.testing.assert_allclose(Y_rot.flatten(), [0.0, 0.0, 2.0], atol=1e-7)

def test_rotation_arbitrary_axis():
    # Fix the random seed for reproducibility
    np.random.seed(42)
    
    for _ in range(10):
        # Generate two random points in 3D
        pt1 = np.random.uniform(-10.0, 10.0, 3)
        pt2 = np.random.uniform(-10.0, 10.0, 3)
        # Ensure they are not too close
        while np.linalg.norm(pt1 - pt2) < 1.0:
            pt2 = np.random.uniform(-10.0, 10.0, 3)
            
        A = homogenize(pt1.reshape(-1, 1))
        B = homogenize(pt2.reshape(-1, 1))
        
        # Plücker line
        L = pluecker.join_points(A, B)
        
        # Random rotation angle
        angle = np.random.uniform(-np.pi, np.pi)
        
        # Compute the rotation matrix from Plücker coordinates
        T = rotation(L, angle)
        
        # Verify it's a valid homogeneous transformation (rigid motion)
        # Check last row is [0, 0, 0, 1]
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-7)
        # Check upper-left 3x3 is orthogonal
        R_mat = T[:3, :3]
        np.testing.assert_allclose(R_mat.T @ R_mat, np.eye(3), atol=1e-7)
        np.testing.assert_allclose(np.linalg.det(R_mat), 1.0, atol=1e-7)
        
        # Let's verify on a random point Q
        q = np.random.uniform(-10.0, 10.0, 3)
        Q = homogenize(q.reshape(-1, 1))
        
        # Rotate Q using T
        Q_rot = dehomogenize(T @ Q).flatten()
        
        # Expected rotation:
        # Direction of rotation axis
        d = pluecker.direction(L).flatten()
        axis = d / np.linalg.norm(d)
        
        # Point on the axis (e.g. closest point to origin)
        p = dehomogenize(pluecker.closest_point_to_origin(L)).flatten()
        
        # Compute expected rotation of Q about axis passing through P
        rot = R.from_rotvec(angle * axis)
        expected_q_rot = rot.apply(q - p) + p
        
        np.testing.assert_allclose(Q_rot, expected_q_rot, atol=1e-7)

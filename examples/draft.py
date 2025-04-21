import numpy as np


def main():
    points = np.array([
        [0, 0],
        [0, 1],
        [-0.5, 0.5],
        [-0.5, -0.5],
        [0.6, -0.5],
        [0.4, 0.5],
        [0.4, 1.2],
        [1.3, 1.1],
        [2, 0.3],
        [-0.5, 1.8],
        [-1, 1.5],
        [-1.3, 1.5],
        [-2, 1.3],
        [-1.5, 1.8],
        [-2.1, 2],
    ])
    
    count = 0
    for dim in range(2, 9):
        if points.shape[1] == dim:
            points_to_check = points
        else:
            points_to_check = np.zeros((points.shape[0], dim))
        if dim % 2 == 0:
            p1 = np.array([1]*(int(dim/2)) + [0]*(int(dim/2)))
            p2 = np.array([0]*(int(dim/2)) + [1]*(int(dim/2)))
            points_to_check[:, 0:int(dim/2)] = np.concatenate(
                [
                    (points[:, 0]/int(dim/2)).reshape((points.shape[0], 1))
                ]*int(dim/2),
                axis=1
            )
            points_to_check[:, int(dim/2):] = np.concatenate(
                [
                    (points[:, 1]/int(dim/2)).reshape((points.shape[0], 1))
                ]*int(dim/2),
                axis=1
            )
        else:
            p1 = np.array([1]*(int((dim+1)/2)) + [0]*(int((dim-1)/2)))
            p2 = np.array([0]*(int((dim+1)/2)) + [1]*(int((dim-1)/2)))
            points_to_check[:, 0:int((dim+1)/2)] = np.concatenate(
                [
                    (points[:, 0]/int((dim+1)/2)).reshape((points.shape[0], 1))
                ]*int((dim+1)/2), 
                axis=1
            )
            points_to_check[:, int((dim+1)/2):] = np.concatenate(
                [
                    (points[:, 1]/int((dim-1)/2)).reshape((points.shape[0], 1))
                ]*int((dim-1)/2), 
                axis=1
            )
        
        for point in points_to_check:
            if (-1-np.dot(point, p1)) >= 0 and (1-np.dot(point, p2)) >= 0:
                print(f"Point {point} is in the unsafe region of dimension {dim}")
                count += 1
            else:
                print(f"Point {point} is not in the unsafe region of dimension {dim}")
    

if __name__ == "__main__":
    main()

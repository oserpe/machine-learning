from ..models.Metrics import Metrics
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def plot_n_k_fold_cv_eval(x, y, n, model, k: int, x_column_names: list = None, y_column_names: list = None):
    # Get ImageClasses possible values
    model.classes = [0,1,2]
    avg_metrics, std_metrics = Metrics.n_k_fold_cross_validation_eval(x, y, n, model, k, x_column_names, y_column_names)
    Metrics.plot_metrics_heatmap_std(avg_metrics, std_metrics, plot_title=f'K Fold Cross Validation Evaluation')


def plot_data_3d(X, Y, Z, point_class):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(X, Y, Z, c=point_class, cmap='jet');

    # Set the axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Rotate the axes and update
    for angle in range(0, 360*4 + 1):
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180

        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        elev = azim = roll = 0
        azim = angle_norm

        # Update the axis view and title
        ax.view_init(elev, azim, roll)
        plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

        plt.draw()
        plt.pause(.001)
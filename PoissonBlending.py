from abc import ABC, abstractmethod
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import napari

DERIVATIVE_KERNEL = [-1, 1, 0]
FORWARD_DERIVATIVE_KERNEL = [-1, 1]
LINEAR_SYSTEM_ALREADY_BUILT_ERROR = "Linear system already built for this blender"
LINEAR_SYSTEM_NOT_BUILT_ERROR = "Linear system has to be built before solving. Call self.build_linear_system"
ALREADY_BLENDED_ERROR = "Images already blended for this blender"
NOT_BLENDED_ERROR = "Images have to be blended before showing results. Call self.blend"
NDIM_MISMATCH_ERROR = ("Invalid dimensions for a {0}D blender: Mask must be {0}D. Source and target must be {0}D "
                       "images, possibly with an extra dimension for channels (either in both or in neither).\n"
                       "Dimensions got: mask.dim={1}, source.ndim={2}, target.ndim={3}")
MASK_SHAPE_ERROR = "mask.shape {} does not match source.shape {}"
SOURCE_NOT_CONTAINED_ERROR = "source.shape {} has to be contained in target.shape {} in all dimensions"


class PoissonBlendingException(Exception):
    pass


def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Linearly rescales an array to the [0, 1] range.
    :param arr: input array
    :return: rescaled array
    """
    arr_max, arr_min = np.max(arr), np.min(arr)
    return (arr - arr_min) / (arr_max - arr_min)


class PoissonBlender(ABC):
    """
    An abstract base class for Poisson blending of images.
    """
    NDIM = None

    def __init__(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray, mix_gradients: bool = True) -> None:
        """
        Constructs a PoissonBlender object.
        :param source: image to blend into the target
        :param target: image to blend the source into
        :param mask: a binary image indicating which source pixels to blend. shape has to match source
        :param mix_gradients: whether to perform gradient mixing, enhancing results for blending images of partially
                              transparent objects. Defaults to True.
        """
        self.source = source
        self.target = target
        self.mask = mask
        self.mix_gradients = mix_gradients
        self._validate_and_preprocess_inputs()
        self.region_idx = np.flatnonzero(self.mask)

        self.A, self.b = None, None
        self.blended = None

    def _validate_and_preprocess_inputs(self) -> None:
        """
        Performs input validation and in-place preprocessing. Raises ValueError.
        """
        # ndim validation
        if not self.mask.ndim == self.NDIM <= self.source.ndim == self.target.ndim <= self.NDIM + 1:
            raise ValueError(NDIM_MISMATCH_ERROR.format(self.NDIM, self.mask.ndim, self.source.ndim, self.target.ndim))

        # mask shape validation
        if self.mask.shape != self.source.shape[:self.NDIM]:
            raise ValueError(MASK_SHAPE_ERROR.format(self.mask.shape, self.source.shape))

        # normalize intensity
        self.source = min_max_normalize(self.source)
        self.target = min_max_normalize(self.target)

        # enforce binary mask
        self.mask = np.round(self.mask)

        # pad source to target shape
        shapes_half_diff = (np.array(self.target.shape) - np.array(self.source.shape)) / 2
        if any(shapes_half_diff < 0):  # source contained in target validation
            raise ValueError(SOURCE_NOT_CONTAINED_ERROR.format(self.source.shape, self.target.shape))

        pad_widths = tuple(zip(np.floor(shapes_half_diff).astype(int), np.ceil(shapes_half_diff).astype(int)))
        self.source = np.pad(self.source, pad_widths)

        # pad mask, which may of a lower dimension if using colored source and target
        self.mask = np.pad(self.mask, pad_widths[:self.NDIM])

    def blend(self) -> np.ndarray:
        """
        Blends the source image into the target image according to the mask.
        :return: blended image
        """
        if self.blended is not None:
            raise PoissonBlendingException(ALREADY_BLENDED_ERROR)

        self._build_linear_system()
        solution = self._solve_linear_system().clip(0, 1)
        self.blended = self.target.copy()
        self.blended[self.mask.astype(bool)] = solution
        return self.blended

    def _solve_linear_system(self) -> np.ndarray:
        """
        Solves the Poisson equations for this image blending problem. Requires _build_linear_system to have been called.
        :return: optimal pixel values for the blended region
        """
        if self.A is None or self.b is None:
            raise PoissonBlendingException(LINEAR_SYSTEM_NOT_BUILT_ERROR)

        return scipy.sparse.linalg.spsolve(self.A, self.b)

    def _build_linear_system(self) -> None:
        """
        Updates the internal state this blender with the Poisson equations for this image blending problem.
        """
        if self.A is not None or self.b is not None:
            raise PoissonBlendingException(LINEAR_SYSTEM_ALREADY_BUILT_ERROR)

        boundary = find_boundaries(self.mask, mode="inner").astype(int)
        boundary_idx = np.flatnonzero(boundary)
        interior = self.mask - boundary
        interior_idx = np.flatnonzero(interior)

        interior_pos = np.searchsorted(self.region_idx, interior_idx)
        boundary_pos = np.searchsorted(self.region_idx, boundary_idx)

        A = scipy.sparse.lil_array((len(self.region_idx), len(self.region_idx)))
        neighbors_positions = self.get_neighbors_positions(interior_idx, self.region_idx, self.mask.shape)
        for neighbors_pos in neighbors_positions:
            A[interior_pos, neighbors_pos] = 1
        A[interior_pos, interior_pos] = -len(neighbors_positions)
        A[boundary_pos, boundary_pos] = 1
        self.A = A.tocsr()

        laplacian = self.get_laplacian(self.mix_gradients)
        interior_laplacians = laplacian[interior.astype(bool)]
        boundary_conditions = self.target[boundary.astype(bool)]
        self.b = np.zeros(len(self.region_idx))
        self.b[interior_pos] = interior_laplacians
        self.b[boundary_pos] = boundary_conditions

    @staticmethod
    @abstractmethod
    def get_neighbors_positions(idx: np.ndarray, region_idx: np.ndarray, shape: tuple) -> tuple:
        """
        Given a region in an array and indices within it, finds the positions of the neighbors of those indices,
        relative to the region.
        For example, if the array has shape (5, ), and the region of interest is indices [1, 2, 3], the neighbors of
        index 2 are in positions 0 and 2 inside the region.
        :param idx: indices whose neighbors to find, counted as if the array is flattened. This param is a 1D array.
        :param region_idx: indices of a region, counted as if the array is flattened. This param is a 1D array.
        :param shape: the shape of the array
        :return: a tuple where each element is an ndarray with shape (len(idx), ) containing the neighbors for the
        given indices in a specific direction. There are two neighbors in each dimension, hence the length of the
        returned tuple is 2*len(shape).
        """
        pass

    @abstractmethod
    def get_laplacian(self, mix_gradients: bool) -> np.ndarray:
        """
        Finds the laplacian of the source image.
        :param mix_gradients: whether to perform gradient mixing, meaning that before taking a second derivative to
        compute the laplacian, the gradient of the source image is replaced by that of the target image in pixels where
        the latter has a larger magnitude.
        :return: ndarray
        """
        pass

    @abstractmethod
    def show_results(self) -> plt.Figure:
        """
        Visualizes the blended image, alongside the blender's inputs. Requires blend to have been called.
        :return: Figure object
        """
        pass


class Poisson2DBlender(PoissonBlender):
    """
    Implements the abstract methods of the PoissonBlender interface for 2D grayscale images.
    """
    NDIM = 2

    # 2D reshaping patterns for derivative kernels, used in get_gradient:
    X_SHAPE = (1, -1)
    Y_SHAPE = (-1, 1)

    @staticmethod
    def get_neighbors_positions(idx: np.ndarray, region_idx: np.ndarray, shape: tuple) -> tuple:
        """
        2D implementation of the abstract method defined in PoissonBlender. See docs there.
        """
        region_width = shape[-1]

        n1_pos = np.searchsorted(region_idx, idx - 1)  # left neighbors
        n2_pos = np.searchsorted(region_idx, idx + 1)  # right neighbors
        n3_pos = np.searchsorted(region_idx, idx - region_width)  # up neighbors
        n4_pos = np.searchsorted(region_idx, idx + region_width)  # down neighbors

        return n1_pos, n2_pos, n3_pos, n4_pos

    def get_gradient(self, arr: np.ndarray, forward: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the gradient of a given array.
        :param arr: 2D np.ndarray
        :param forward: whether to use forward derivative kernels. Defaults to True.
        :return: a 2-tuple with the (x, y) directions of the gradient, each an ndarray with shape like arr.
        """
        kernel = FORWARD_DERIVATIVE_KERNEL if forward else DERIVATIVE_KERNEL
        kx = np.reshape(kernel, self.X_SHAPE)
        ky = np.reshape(kernel, self.Y_SHAPE)
        Gx = scipy.signal.fftconvolve(arr, kx, mode="same")
        Gy = scipy.signal.fftconvolve(arr, ky, mode="same")
        return Gx, Gy

    def get_laplacian(self, mix_gradients: bool) -> np.ndarray:
        """
        2D implementation of the abstract method defined in PoissonBlender. See docs there.
        """
        Gx_src, Gy_src = self.get_gradient(self.source)
        Gx_target, Gy_target = self.get_gradient(self.target)
        G_src_squared_magnitude = Gx_src ** 2 + Gy_src ** 2
        G_target_squared_magnitude = Gx_target ** 2 + Gy_target ** 2
        if mix_gradients:
            Gx = np.where(G_src_squared_magnitude > G_target_squared_magnitude, Gx_src, Gx_target)
            Gy = np.where(G_src_squared_magnitude > G_target_squared_magnitude, Gy_src, Gy_target)
        else:
            Gx, Gy = Gx_src, Gy_src

        Gxx, _ = self.get_gradient(Gx, forward=False)
        _, Gyy = self.get_gradient(Gy, forward=False)
        laplacian = Gxx + Gyy
        return laplacian

    @staticmethod
    def plot_2d_results(source, mask, target, blended) -> plt.Figure:
        fig = plt.figure(layout="constrained")
        inputs_fig, blended_fig = fig.subfigures(nrows=2)
        input_axs = inputs_fig.subplots(ncols=3)
        blended_ax = blended_fig.subplots()
        axes = list(input_axs) + [blended_ax]
        images = source, mask, target, blended
        titles = ("source", "mask", "target", "blended")
        for img, title, ax in zip(images, titles, axes):
            ax.imshow(img, cmap="gray")
            ax.set(title=title, xticks=[], yticks=[])

        return fig

    def show_results(self) -> plt.Figure:
        """
        2D implementation of the abstract method defined in PoissonBlender. See docs there.
        """
        if self.blended is None:
            raise PoissonBlendingException(NOT_BLENDED_ERROR)

        fig = self.plot_2d_results(self.source, self.mask, self.target, self.blended)
        plt.show()
        return fig


class Poisson3DBlender(PoissonBlender):
    """
    Implements the abstract methods of the PoissonBlender interface for 3D grayscale images.
    """
    NDIM = 3

    # 3D reshaping patterns for derivative kernels, used in get_gradient:
    X_SHAPE = (1, 1, -1)
    Y_SHAPE = (1, -1, 1)
    Z_SHAPE = (-1, 1, 1)

    @staticmethod
    def get_neighbors_positions(idx: np.ndarray, region_idx: np.ndarray, shape: tuple) -> tuple:
        """
        3D implementation of the abstract method defined in PoissonBlender. See docs there.
        """
        region_height, region_width = shape[-2:]

        n1_pos, n2_pos, n3_pos, n4_pos = Poisson2DBlender.get_neighbors_positions(idx, region_idx, shape)
        n5_pos = np.searchsorted(region_idx, idx - region_height * region_width)  # previous plane neighbors
        n6_pos = np.searchsorted(region_idx, idx + region_height * region_width)  # next plane neighbors

        return n1_pos, n2_pos, n3_pos, n4_pos, n5_pos, n6_pos

    def get_gradient(self, arr: np.ndarray, forward: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Finds the gradient of a given array.
        :param arr: 3D np.ndarray
        :param forward: whether to use forward derivative kernels. Defaults to True.
        :return: a 3-tuple with the (x, y, z) directions of the gradient, each an ndarray with shape like arr.
        """
        kernel = FORWARD_DERIVATIVE_KERNEL if forward else DERIVATIVE_KERNEL
        kx = np.reshape(kernel, self.X_SHAPE)
        ky = np.reshape(kernel, self.Y_SHAPE)
        kz = np.reshape(kernel, self.Z_SHAPE)
        Gx = scipy.signal.fftconvolve(arr, kx, mode="same")
        Gy = scipy.signal.fftconvolve(arr, ky, mode="same")
        Gz = scipy.signal.fftconvolve(arr, kz, mode="same")
        return Gx, Gy, Gz

    def get_laplacian(self, mix_gradients: bool) -> np.ndarray:
        """
        3D implementation of the abstract method defined in PoissonBlender. See docs there.
        """
        Gx_src, Gy_src, Gz_src = self.get_gradient(self.source)
        Gx_target, Gy_target, Gz_target = self.get_gradient(self.target)
        G_src_squared_magnitude = Gx_src ** 2 + Gy_src ** 2 + Gz_src ** 2
        G_target_squared_magnitude = Gx_target ** 2 + Gy_target ** 2 + Gz_target ** 2
        if mix_gradients:
            Gx = np.where(G_src_squared_magnitude > G_target_squared_magnitude, Gx_src, Gx_target)
            Gy = np.where(G_src_squared_magnitude > G_target_squared_magnitude, Gy_src, Gy_target)
            Gz = np.where(G_src_squared_magnitude > G_target_squared_magnitude, Gz_src, Gz_target)
        else:
            Gx, Gy, Gz = Gx_src, Gy_src, Gz_src

        Gxx, _, _ = self.get_gradient(Gx, forward=False)
        _, Gyy, _ = self.get_gradient(Gy, forward=False)
        _, _, Gzz = self.get_gradient(Gz, forward=False)
        laplacian = Gxx + Gyy + Gzz
        return laplacian

    def show_results(self) -> plt.Figure:
        """
        3D implementation of the abstract method defined in PoissonBlender. See docs there.
        """
        mid_plane = self.mask.shape[0] // 2
        mid_plane_results = Poisson2DBlender.plot_2d_results(self.source[mid_plane], self.mask[mid_plane], self.target[
            mid_plane], self.blended[mid_plane])
        plt.suptitle("Mid cross-section")
        plt.show()

        viewer = napari.Viewer()
        viewer.dims.ndisplay = 3
        images = self.source, self.mask, self.target, self.blended
        titles = ("source", "mask", "target", "blended")
        for img, title in zip(images, titles):
            viewer.add_image(img, name=title)
        napari.run()

        return mid_plane_results


class Colored(PoissonBlender):
    """
    A decorator class for PoissonBlenders. Adds multichannel (i.e., RGB) support.
    """

    def __init__(self, blender: PoissonBlender) -> None:
        self.blender = blender
        self.source = blender.source.copy()
        self.target = blender.target.copy()
        self.blended = None

    def blend(self) -> np.ndarray:
        """
        A multichannel implementation of the blending procedure, based on the single-channel capability of the
        decorated blender.
        """
        blended_channels = []
        for channel in range(self.source.shape[-1]):  # channel-wise blending
            # reset internal state for next channel blending:
            self.blender.source = self.source[..., channel]
            self.blender.target = self.target[..., channel]
            self.blender.A, self.blender.b = None, None
            self.blender.blended = None

            blended_channel = self.blender.blend()
            blended_channels.append(blended_channel)

        self.blended = np.dstack(blended_channels)
        self.blender.source = self.source
        self.blender.target = self.target
        self.blender.blended = self.blended
        return self.blended

    def get_laplacian(self, mix_gradients: bool) -> np.ndarray:
        return self.blender.get_laplacian(mix_gradients)

    def get_neighbors_positions(self, idx: np.ndarray, region_idx: np.ndarray, shape: tuple) -> tuple:
        return self.blender.get_neighbors_positions(idx, region_idx, shape)

    def show_results(self) -> plt.Figure:
        return self.blender.show_results()

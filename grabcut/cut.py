"""This module contains logic for the grabcut algorithm."""
# import cv2
import numpy as np
import maxflow
# from graph import Graph
# cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)


'''
need 2 GMMs: foreground and background
'''


class GMM:
    """This class defines a gaussian mixture model."""

    def __init__(self, k=5):
        """Initialize with a default of k = 5."""
        self.k = k
        self.means = np.zeros((k, 3))
        self.cov = np.zeros((k, 3, 3))
        self.inv_cov = np.zeros((k, 3, 3))
        self.det_cov = np.zeros((k, 1))
        self.weights = np.zeros((k, 1))

        self.total_pixel_count = 0

        self.eigenvalues = np.zeros(k)
        self.eigenvectors = np.zeros((k, 3))
        self.pixels = [[] for _ in range(k)]

    def add_pixel(self, pixel, i):
        """Add a pixel to the GMM."""
        self.pixels[i].append(pixel.copy())
        self.total_pixel_count += 1

    def update_gmm(self):
        """Update the means and covs for the GMM."""
        for i in range(self.k):
            n = len(self.pixels[i])
            if n == 0:
                self.weights[i] = 0
            else:
                self.weights[i] = n / self.total_pixel_count
                self.means[i] = np.mean(self.pixels[i], axis=0)

                self.cov[i] = np.cov(self.pixels[i], rowvar=False, bias=True)
                # print("cov ", i, self.cov[i])
                self.det_cov[i] = np.linalg.det(self.cov[i])
                while self.det_cov[i] <= 0:
                    self.cov[i] += np.diag([0.1, 0.1, 0.1])
                    self.det_cov = np.linalg.det(self.cov[i])
                self.inv_cov[i] = np.linalg.inv(self.cov[i])

                evals, evects = np.linalg.eig(self.cov[i])
                max_ind = np.argmax(evals)
                self.eigenvalues[i] = evals[max_ind]
                self.eigenvectors[i] = evects[max_ind]

    '''
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EITHER THIS ONE OR THE ONE BELOW WORKS
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''

    def redistribute_pixels(self):
        """Redistribute the pixels to different clusters."""
        self.update_gmm()
        for i in range(1, self.k):
            n = np.argmax(self.eigenvalues)
            e_n = self.eigenvectors[n]
            rhs = np.dot(e_n.T, self.means[n])
            lhs = np.dot(e_n.T, np.array(self.pixels[n]).T)
            # print("lhs: ", lhs)
            # print("rhs: ", rhs)
            # e_n = np.tile(e_n, (len(self.pixels[n]), 1))
            indices1 = np.where(lhs <= rhs)
            indices2 = np.where(lhs > rhs)
            # print(indices1)
            # print(indices2)
            # print(self.pixels[n])
            temp = np.asarray(self.pixels[n])
            self.pixels[i] = [p for p in temp[indices1]]
            self.pixels[n] = [p for p in temp[indices2]]
            self.update_gmm()

    def redistribute_all_pixels(self):
        """Redistribute the pixels to different clusters."""
        self.update_gmm()
        for i in range(self.k):
            n = np.argmax(self.eigenvalues)
            e_n = self.eigenvectors[n]
            rhs = np.dot(e_n.T, self.means[n])
            lhs = np.dot(e_n.T, np.array(self.pixels[n]).T)
            # print("lhs: ", lhs)
            # print("rhs: ", rhs)
            # e_n = np.tile(e_n, (len(self.pixels[n]), 1))
            indices1 = np.where(lhs <= rhs)
            indices2 = np.where(lhs > rhs)
            # print(indices1)
            # print(indices2)
            # print(self.pixels[n])
            temp = np.asarray(self.pixels[n])
            self.pixels[i] = [p for p in temp[indices1]]
            self.pixels[n] = [p for p in temp[indices2]]
            self.update_gmm()

    def calculate_values_from_hardcoded(self):
        """Calculate det and inv from hardcoded values."""
        for i in range(self.k):
            self.det_cov[i] = np.linalg.det(self.cov[i])
            while self.det_cov[i] <= 0:
                self.cov[i] += np.diag([0.1, 0.1, 0.1])
                self.det_cov = np.linalg.det(self.cov[i])
            self.inv_cov[i] = np.linalg.inv(self.cov[i])

    @staticmethod
    def load_gmm_from_values(weight_vals, mean_vals, covar_vals):
        """Load a gmm from some hardcoded values."""
        assert(len(weight_vals) is 5)
        assert(len(mean_vals) is 15)
        assert(len(covar_vals) is 45)
        gmm = GMM()
        for i in range(5):
            gmm.weights[i] = weight_vals[i]
            gmm.means[i] = mean_vals[3 * i:3 * i + 3]
            gmm.cov[i] = covar_vals[9 * i:9 * i + 9].reshape((3, 3))
        gmm.calculate_values_from_hardcoded()
        return gmm


class GrabCut:
    """This class represents the engine for grabcut."""

    def __init__(self, img, k=5):
        """Initialize the object with an image."""
        self.img = img
        self.k = k
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.trimap = np.zeros((self.height, self.width))
        self.matte = np.zeros((self.height, self.width))
        self.comp_index = np.zeros((self.height, self.width))
        self.d_fgd = np.zeros((self.height, self.width))
        self.d_bgd = np.zeros((self.height, self.width))
        self.background_gmm = None
        self.foreground_gmm = None
        self.first_iteration_complete = False
        self.bg = 0
        self.fg = 1
        self.pr_bg = 2
        self.pr_fg = 3

    '''
    step 1
    '''
    def convert_rect_to_mask(self, rect, img):
        """Convert a rect to a trimap mask."""
        mask = np.zeros((img.shape[0], img.shape[1]))
        mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = self.pr_fg
        return mask

    def convert_rect_to_matte(self, rect, img):
        """Convert a rect to a matte."""
        matte = np.zeros((img.shape[0], img.shape[1]))
        matte[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 1
        return matte

    def set_bgd_fgd(self):
        """Set the background and foreground pixel sets."""
        self.bgd = np.where(np.logical_or(self.trimap == self.bg, self.trimap == self.pr_bg))
        self.fgd = np.where(np.logical_or(self.trimap == self.fg, self.trimap == self.pr_fg))
        self.bgd_pixels = self.img[self.bgd]
        self.fgd_pixels = self.img[self.fgd]

    def prob_pixel_in_gmm(self, pixel, model):
        """Calculate the probability a pixel is in a model, and the cluster index that it would belong to."""
        prob_vals = []
        s = 0
        for i in range(model.k):
            inv = model.inv_cov[i]
            det = model.det_cov[i]

            diff = pixel - model.means[i]
            diff = np.asarray([diff])
            m = np.dot(inv, diff.T)
            m = np.dot(diff, m)
            # print(inv, diff, m)
            m = (np.exp(-0.5 * m) / np.sqrt(det))[0][0]
            prob_vals.append(m)
            m *= model.weights[i]
            s += m
        ind = np.argmax(np.asarray(prob_vals))
        # print(-np.log(s)[0])
        return -np.log(s)[0], ind

    def get_beta(self):
        """Get the beta value based on the paper."""
        left_diffs = self.img[:, 1:] - self.img[:, :-1]
        upleft_diffs = self.img[1:, 1:] - self.img[:-1, :-1]
        up_diffs = self.img[1:, :] - self.img[:-1, :]
        upright_diffs = self.img[1:, :-1] - self.img[:-1, 1:]
        sum_squared = (left_diffs * left_diffs).sum() + (upleft_diffs * upleft_diffs).sum() + \
                      (up_diffs * up_diffs).sum() + (upright_diffs * upright_diffs).sum()
        beta = sum_squared / (4 * self.img.shape[0] * self.img.shape[1] - 3 * (self.img.shape[0] + self.img.shape[1]) + 2)
        return 1 / (2 * beta)

    def build_n_link(self, nodeids):
        """Build the neighbour links."""
        diag_left = np.zeros((self.img.shape[0], self.img.shape[1]))
        diag_right = np.zeros((self.img.shape[0], self.img.shape[1]))
        up = np.zeros((self.img.shape[0], self.img.shape[1]))
        left = np.zeros((self.img.shape[0], self.img.shape[1]))

        beta = self.get_beta()

        for y in range(self.height):
            for x in range(self.width):
                z_m = self.img[y][x]
                # node_id = nodeids[y][x]
                self.max_weight = float('-inf')
                if y > 0 and x > 0:
                    diff = (z_m - self.img[y - 1][x - 1])
                    diag_left[y][x] = 50 / np.sqrt(2) * np.exp(-beta * np.dot(diff, diff))
                    if diag_left[y][x] > self.max_weight:
                        self.max_weight = diag_left[y][x]
                    # diag_left_id = nodeids[y - 1][x - 1]
                    # self.graph.add_edge(node_id, diag_left_id, diag_left[y][x], diag_left[y][x])
                if y > 0 and x < self.width - 1:
                    diff = (z_m - self.img[y - 1][x + 1])
                    diag_right[y][x] = 50 / np.sqrt(2) * np.exp(-beta * np.dot(diff, diff))
                    if diag_right[y][x] > self.max_weight:
                        self.max_weight = diag_right[y][x]
                    # diag_right_id = nodeids[y - 1][x + 1]
                    # self.graph.add_edge(node_id, diag_right_id, diag_right[y][x], diag_right[y][x])
                if x > 0:
                    diff = (z_m - self.img[y][x - 1])
                    left[y][x] = 50 * np.exp(-beta * np.dot(diff, diff))
                    if left[y][x] > self.max_weight:
                        self.max_weight = left[y][x]
                    # left_id = nodeids[y][x - 1]
                    # self.graph.add_edge(node_id, left_id, left[y][x], left[y][x])
                if y > 0:
                    diff = (z_m - self.img[y - 1][x])
                    up[y][x] = 50 * np.exp(-beta * np.dot(diff, diff))
                    if up[y][x] > self.max_weight:
                        self.max_weight = up[y][x]
                    # up_id = nodeids[y - 1][x]
                    # self.graph.add_edge(node_id, up_id, up[y][x], up[y][x])

        diag_left_struct = np.array([[1, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 0]])
        diag_right_struct = np.array([[0, 0, 1],
                                      [0, 0, 0],
                                      [0, 0, 0]])
        up_struct = np.array([[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]])
        left_struct = np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 0, 0]])
        self.graph.add_grid_edges(nodeids, weights=diag_left, structure=diag_left_struct,
                                  symmetric=True)
        self.graph.add_grid_edges(nodeids, weights=diag_right, structure=diag_right_struct,
                                  symmetric=True)
        self.graph.add_grid_edges(nodeids, weights=left, structure=left_struct,
                                  symmetric=True)
        self.graph.add_grid_edges(nodeids, weights=up, structure=up_struct,
                                  symmetric=True)

    def build_t_link(self, nodeids):
        """Build the target links."""
        for y in range(self.height):
            for x in range(self.width):
                if self.trimap[y][x] == self.bg:
                    self.graph.add_tedge(nodeids[y][x], self.max_weight, 0)
                elif self.trimap[y][x] == self.fg:
                    self.graph.add_tedge(nodeids[y][x], 0, self.max_weight)
                else:
                    d_f = self.d_fgd[y][x]
                    d_b = self.d_bgd[y][x]
                    # print(d_b, d_f)
                    self.graph.add_tedge(nodeids[y][x], d_f, d_b)

    def update_gmm_components(self):
        """Update self.comp_index."""
        self.comp_index = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                df, ind_f = self.prob_pixel_in_gmm(self.img[y][x], self.foreground_gmm)
                db, ind_b = self.prob_pixel_in_gmm(self.img[y][x], self.background_gmm)
                if self.trimap[y][x] == self.bg or self.trimap[y][x] == self.pr_bg:
                    self.comp_index[y][x] = ind_f
                else:
                    self.comp_index[y][x] = ind_b
                self.d_fgd[y][x] = df
                self.d_bgd[y][x] = db

    def update_trimap_from_segmentation(self, segmentation):
        """Update the trimap based on the segmentation."""
        for y in range(self.height):
            for x in range(self.width):
                if self.trimap[y][x] == self.pr_bg or self.trimap[y][x] == self.pr_fg:
                    if segmentation[y][x]:
                        self.trimap[y][x] = self.pr_fg
                    else:
                        self.trimap[y][x] = self.pr_bg

    def update_trimap_from_mask(self, mask):
        """Update the trimap based on the mask."""
        # This is its own function because it might need more processing in the future
        # Idk
        for y in range(self.height):
            for x in range(self.width):
                self.trimap[y][x] = mask[y][x]

    def grab_cut(self, img, mask, rect, use_mask, bgd_gmm=None, fgd_gmm=None):
        """Perform an iteration of grabcut."""
        print("Starting grabcut.")
        if not use_mask:
            self.trimap = self.convert_rect_to_mask(rect, img)
            self.matte = self.convert_rect_to_matte(rect, img)
        else:
            self.update_trimap_from_mask(mask)

        self.set_bgd_fgd()

        if bgd_gmm is None:
            # if not self.first_iteration_complete:
            self.background_gmm = GMM()

            for pixel in self.bgd_pixels:
                self.background_gmm.add_pixel(pixel, 0)

            # if not self.first_iteration_complete:
            self.background_gmm.redistribute_pixels()
            # else:
            #     self.background_gmm.redistribute_all_pixels()
        else:
            self.background_gmm = bgd_gmm

        if fgd_gmm is None:
            # if not self.first_iteration_complete:
            self.foreground_gmm = GMM()

            for pixel in self.fgd_pixels:
                self.foreground_gmm.add_pixel(pixel, 0)
            self.foreground_gmm.redistribute_pixels()
        else:
            self.foreground_gmm = fgd_gmm

        self.update_gmm_components()
        print("Updated GMM components")

        # build the graph
        self.graph = maxflow.Graph[float]()
        # self.graph = Graph()
        nodeids = self.graph.add_grid_nodes(
            (self.img.shape[0], self.img.shape[1]))
        print("Created graph")
        # self.update_gmm_components()
        self.build_n_link(nodeids)
        self.build_t_link(nodeids)
        print("Added weights")
        self.graph.maxflow()
        print("Finished maxflow")

        self.first_iteration_complete = True

        sgm = self.graph.get_grid_segments(nodeids).astype(np.uint32)
        sgm = np.bitwise_and(sgm, self.matte.astype(np.uint32))
        self.update_trimap_from_segmentation(sgm)
        mask = self.trimap.copy()
        return self.trimap

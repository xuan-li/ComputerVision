import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay
import sys

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=20)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)



# ================== GUI ===================

def mouse(event,x,y,flags,param):
    
    global l_button_down
    global mode

    if event == cv2.EVENT_LBUTTONDOWN:
        l_button_down = True
    if event == cv2.EVENT_LBUTTONUP:
        l_button_down = False
        recalculate_mask()

    elif l_button_down:
        draw_circle(x,y,mode)

def draw_circle(x,y, mode = 0):
    radius = 6
    if mode == 0:
        cv2.circle(input_mask,(x,y),radius,(0,0,255),-1)
        cv2.circle(img,(x,y),radius,(0,0,255),-1)
    elif mode == 1:
        cv2.circle(input_mask,(x,y),radius,(255,0,0),-1)
        cv2.circle(img,(x,y),radius,(255,0,0),-1)


def reset():
    global img
    global input_mask
    global output_mask
    global output_img
    print "reset"
    img = original_img.copy()
    input_mask = np.full(img.shape, 255, dtype = np.uint8)
    output_mask = np.full(input_mask.shape[0:2], 255, dtype = np.uint8)
    output_img = original_img.copy()

def recalculate_mask():
    print "compute mask"
    global output_mask
    global output_img
    fg_segments, bg_segments = find_superpixels_under_marking(input_mask, superpixels)
    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)
    norm_hists = normalize_histograms(color_hists)
    fgbg_hists = [fg_cumulative_hist, bg_cumulative_hist]
    fgbg_superpixels = [fg_segments, bg_segments]
    graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)
    output_mask = np.zeros(input_mask.shape[0:2], dtype = np.uint8)
    for i in range(graph_cut.shape[0]):
        output_mask[superpixels == i] = 255 if graph_cut[i] else 0
    output_img = original_img.copy()
    output_img[output_mask == 0] = 0


def main():
    # flags
    print "Press F, then start to draw red lines\nPress B, then start to draw blue lines\nPress M to convert between mask and image\nPress C to clear all lines"
    global img
    global input_mask
    global output_mask
    global output_img
    global centers, color_hists, superpixels, neighbors
    global mode
    global l_button_down, r_button_down
    global original_img

    mode = 0
    l_button_down = False
    m_button_down = False
    show_foreground = True

    original_img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(original_img)


    img = original_img.copy()
    input_mask = np.full(img.shape, 255, dtype = np.uint8)
    output_mask = np.full(input_mask.shape[0:2], 255, dtype = np.uint8)
    output_img = original_img.copy()

    cv2.namedWindow('Image')
    cv2.namedWindow('Segment')
    cv2.setMouseCallback('Image',mouse)

    while(1):
        cv2.imshow('Image',img)
        if(show_foreground):
            cv2.imshow('Segment',output_img)
        else:
            cv2.imshow('Segment',output_mask)
        c = cv2.waitKey(33)
        if c & 0xFF == 27:
            break
        elif c == 102:
            mode = 0
        elif c == 98:
            mode = 1
        elif c == 99:
            reset()
        elif c == 109:
            show_foreground = not show_foreground

    cv2.destroyAllWindows()

# Create a black image, a window and bind the function to window
if __name__ == '__main__':
    main()




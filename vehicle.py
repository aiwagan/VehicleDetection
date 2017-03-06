
import numpy as np
import cv2
import glob,time
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from collections import deque
from scipy.ndimage.measurements import label

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


class CVehicleDetector:

    def __init__(self):
        self.color_space = 'YCrCb'
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = 'ALL'
        self.spatial_size = (16,16)
        self.hist_bins = 32
        self.numscales = 4
        self.scale=[1.0, 1.25, 1.5, 1.75]
        self.y_start_stop = [[400,496], [400,528], [400,592], [400,656]]
        self.heat_threshold = 8
        self.scaler = StandardScaler()
        self.classifier = LinearSVC(C=1.0,  penalty='l2', loss='hinge') #,dual=False) #loss='hinge',

        #self.classifier = RandomForestClassifier(n_estimators=20, n_jobs=-1)
        self.bbox_buffer = deque(maxlen=10)

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs):
        # Create a list to append feature vectors to

        features = []

        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            # print(file)
            image = mpimg.imread(file)
            cspace = self.color_space
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif cspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                feature_image = np.copy(image)

            # Call get_hog_features() with vis=False, feature_vec=True
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel], self.orient, self.pix_per_cell,
                                        self.cell_per_block, vis=False, feature_vec=True))

                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, self.hog_channel], self.orient, self.pix_per_cell,
                                                self.cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            spatial_features = np.array(bin_spatial(image, size=self.spatial_size))
            hist_features = np.array(color_hist(image, nbins=self.hist_bins))
            hog_features = np.array(hog_features)
            # print(spatial_features)
            # print(hist_features)
            # print(hog_features)

            # features.append(spatial_features)
            # features.append(hist_features)
            # features.append(hog_features)
            # g=hog_features
            g = np.hstack((spatial_features, hist_features, hog_features))
            # print(g.shape)
            features.append(np.ravel(g))

            # Return list of feature vectors()
        return features



    def prepare_classifier(self):
        dirs = ['./data/vehicles/GTI_Far/*.png', './data/vehicles/GTI_Left/*.png',
                './data/vehicles/GTI_MiddleClose/*.png', './data/vehicles/GTI_Right/*.png',
                './data/vehicles/KITTI_extracted/*.png', './data/non-vehicles/Extras/*.png',
                './data/non-vehicles/GTI/*.png']

        # Divide up into cars and notcars
        images = []
        for d in dirs:
            images = images + glob.glob(d)
        # print(images)
        cars = []
        notcars = []
        for image in images:
            # print('Reading:'+image)
            if 'non-vehicles' in image:
                notcars.append(image)
            else:
                cars.append(image)
        # print(notcars)
        # Reduce the sample size because HOG features are slow to compute
        # The quiz evaluator times out after 13s of CPU time
        # sample_size = 5000
        # cars = cars[0:sample_size]
        # notcars = notcars[0:sample_size]

        ### TODO: Tweak these parameters and see how the results change.
        #colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        #orient = 9
        #pix_per_cell = 8
        #cell_per_block = 2
        #hog_channel = "ALL"  # 0 # Can be 0, 1, 2, or "ALL"lf

        t = time.time()

        car_features = self.extract_features(cars)
        notcar_features = self.extract_features(notcars)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        self.scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.3, random_state=rand_state)

        print('Using:', self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        #self.classifier = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        self.classifier.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.classifier.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', self.classifier.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img):
        bbox_list = []
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255
        for i in range(self.numscales):
            curr_scale = self.scale[i]
            start_stop = self.y_start_stop[i]
            img_tosearch = img[start_stop[0]:start_stop[1], :, :]
            ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        #for curr_scale in self.scale:
            if curr_scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / curr_scale), np.int(imshape[0] / curr_scale)))

            ch1 = ctrans_tosearch[:, :, 0]
            ch2 = ctrans_tosearch[:, :, 1]
            ch3 = ctrans_tosearch[:, :, 2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // self.pix_per_cell) - 1
            nyblocks = (ch1.shape[0] // self.pix_per_cell) - 1
            nfeat_per_block = self.orient * self.cell_per_block ** 2
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // self.pix_per_cell) - 1
            cells_per_step = 2  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            # Compute individual channel HOG features for the entire image
            hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            features = []
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb * cells_per_step
                    xpos = xb * cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    xleft = xpos * self.pix_per_cell
                    ytop = ypos * self.pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                    # Get color features
                    spatial_features = bin_spatial(subimg, size=self.spatial_size)
                    hist_features = color_hist(subimg, nbins=self.hist_bins)
                    # print(hist_features.shape)
                    # print(hog_features.shape)
                    # print(spatial_features.shape)
                    # g=np.array([hog_features])
                    # print(g)
                    # g=hog_features
                    # np.ravel(np.hstack((spatial_features, hist_features, hog_features)))
                    # scaler = StandardScaler().fit(g)
                    # print(g.shape)
                    # Scale features and make a prediction
                    # test_features = X_scaler.transform(g)  #.reshape(1, -1)
                    test_features = self.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                    test_prediction = self.classifier.predict(test_features)

                    # print(test_prediction)
                    if test_prediction == 1:
                        xbox_left = np.int(xleft * curr_scale)
                        ytop_draw = np.int(ytop * curr_scale)
                        win_draw = np.int(window * curr_scale)
                        bbox_list.append([(xbox_left, ytop_draw + start_stop[0]),
                                          (xbox_left + win_draw, ytop_draw + win_draw + start_stop[0])])
                        #cv2.rectangle(draw_img, (xbox_left, ytop_draw + self.y_start_stop[0]),
                        #              (xbox_left + win_draw, ytop_draw + win_draw + self.y_start_stop[0]), (0, 0, 255), 6)

            self.bbox_buffer.append(bbox_list)



    def add_heat(self, heatmap):
        # Iterate through list of bboxes
        for bbox_list in self.bbox_buffer:
            for box in bbox_list:
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def apply_threshold(self, heatmap):
        # Zero out pixels below the threshold
        heatmap[heatmap <= self.heat_threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Car:{}'.format(car_number), (bbox[0][0], bbox[0][1]-5), font, 1, (255, 156, 205), 2)
            cv2.rectangle(img, bbox[0], bbox[1], (255, 156, 205), 6)
        # Return the image
        return img

    def process_image(self, img):
        self.find_cars(img)
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        heat = self.add_heat(heat)

        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        draw_img = self.draw_labeled_bboxes(np.copy(img), labels)

        return draw_img




if __name__ == "__main__":

    vdet = CVehicleDetector()
    vdet.prepare_classifier()
    output = './output.mp4'
    clip1 = VideoFileClip('./project_video.mp4')
    # ld = CLaneDetector()
    white_clip = clip1.fl_image(vdet.process_image)
    white_clip.write_videofile(output, audio=False)
    print("process Finished. Please view %s" % output)



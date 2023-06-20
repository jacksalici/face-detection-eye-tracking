# Gaze Detection

> "Robot: ≪Are you looking me?≫"

This repo is part of the Computer Vision course final project. The goal of this first part of the pipeline is to detect faces present in the input of the image stream and analyze each of them to learn if the eyes are gazing into the camera. To achieve this result, we studied a process composed of four different steps: 
1. face detection, 
2. facial landmark detection, 
3. facial pose estimation, and finally
4. precise pupil localization. 

The facial pose estimation is needed to know if the person is facing the camera and, in that case, the pupil localization is computed and we applied a heuristic threshold to approximate whether the gaze is following the face direction. In the next paragraphs, we will discuss the choice of this implementation and why we don't compute the estimation directly from the eye.

## Face detection

To realize the first step, needed not only for the gaze tracking but also for the next parts of the project pipeline, we have studied how to detect faces using one of the many different approaches present in the literature. According to [Zhao et Al, 2003] there are three different models: 1) feature-based, where the goal is to detect distinguished features (like eyes and nose) and compute from their geometrical position if a face is present; 2) template-based, where the input is compared with an already present pattern using an SVM, for instance, and 3) appearance-based, where a small rectangular-shaped patch is overlapped on several windows of the input image. Many different authors have proposed methods based on this approach, with the most cited being the one from [Viola and Jones, 2001].
During our first tests, we used the Haar Cascade classifier that is based on the above-mentioned Viola-Jones. The AdaBoost algorithm is applied over a set of Haar features, the differences between the sum of pixels within adjacent rectangular windows of the image scaled with several factors. We applied the OpenCV library and we tested it detecting faces in Fer2013. We also made the same test using a HOG-based algorithm present in dlib, another C++ library that offers several interesting machine learning and computer vision algorithms. The results of the second method were slightly better as shown in the table.  

|  | No face detected | One face detected | Two faces detected | Time needed to do all the detection |
|---|---|---|---|---|
| Hog Dlib | 10932 | 24955 | 0 | 34.18 s |
| Haar Cascade OpenCv | 15272 | 20608 | 7 | 38.38 s |

_Comparison between two different classical face detection algorithms. They were tested on 35887 images containing a single image. They reach an accuracy of 0,70 and 0,57 respectively. Test made on a 2020 M1 MacBook Air._

Although it may not be the best testing scenario since the faces are often occluded and the test we made not measures important metrics like precision and recall, it has been a very quick way to verify how both methods perform under real-life scenario. We had seen that dlib method handles better occlusions but moreover, is much more rotation invariant than Viola-Jones.

## Facial landmarks detection

For what concern the second step of the process, we get the facial landmarks detection simply using the dlib method. The algorithm is based on a paper from CVPR [Kazemi and Sullivam, 2014] and trained on IBug300-W and it makes use of a cascade ensemble of regression trees to do shape invariant feature selection based on thresholding the diffenrence of intensity values at pixels level. 

The dlib method finds out with a low error 68 landmarks present along the face. From these we only select a few that we have used in the next steps.


![[landmarks.png]]


## Pose estimation

Since the goal of this first part of the pipeline is to detect whether or not a person is facing to the camera, we have to compute the facial pose estimation of the faces detected. We had to solve the so-called Perspective-n-Point (PnP) pose computation problem, computing the rotation and translation $[R|T]$ matrix given the homogeneous coordinates of the face projected on the $[u, v, 1]$ 2d plane, the 3d world face points $p_w$ expressed in world coordinate system and the camera matrix $K$.$$\begin{bmatrix}
  u \\
  v \\
  1
  \end{bmatrix} =
  K \hspace{.3em} \Pi \hspace{.3em}  [R|T]
  \begin{bmatrix}
  X_{w} \\
  Y_{w} \\
  Z_{w} \\
  1
  \end{bmatrix} 
  $$
If expanded, it becomes the following, note that $\Pi$ is the perspective projection model. $$
	\begin{bmatrix}
  u \\
  v \\
  1
  \end{bmatrix} =
  \begin{bmatrix}
  f_x & \gamma & c_x \\
  0 & f_y & c_y \\
  0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
  1 & 0 & 0 & 0 \\
  0 & 1 & 0 & 0 \\
  0 & 0 & 1 & 0
  \end{bmatrix}
  \begin{bmatrix}
  r_{11} & r_{12} & r_{13} & t_x \\
  r_{21} & r_{22} & r_{23} & t_y \\
  r_{31} & r_{32} & r_{33} & t_z \\
  0 & 0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
  X_{w} \\
  Y_{w} \\
  Z_{w} \\
  1
  \end{bmatrix}

$$
While the face points are given from the landmark detection at the previous points, the 3d world point are set using a standard face 3d model. For the best results the 3d points could be measured directly on the target face, but since there is more than a single target, the model is perfectly fitted for a standard human face.

For the camera calibration, we applied the chessboard method implemented in OpenCV, computing the intrinsic camera matrix $K$ starting from several photos of a chessboard. 

To obtain the actual rotation matrix $R$ we used an iterative method  based on a Levenberg-Marquardt optimization present in OpenCV. Then we applied a refinement method also present in the library to get a better prediction. 

Having $R$,  we finally found pitch, roll and yaw of the faces computing the Euler angles from the rotation matrix (get using the Rodrigues formula). 

$$\text{Pitch} = \text{atan2}(r_{32}, r_{33}) $$ $$ \text{Yaw} = \text{atan2}(-r_{31}, \sqrt{r_{32}^2 + R_{33}^2}) $$ $$ \text{Roll} = \text{atan2}(r_{21}, r_{11})$$
![[pose.png]]

Ideally, the face is facing the camera if all three Euler angles are 0. In the real-life scenario, because of the image noise and the not-always-perfect accuracy of the angles found, we had to put a threshold under which the face is considered to be facing. During the testing face, we noticed that a threshold of 20° is fine.

## Precise eye center localization

What if the face is not facing the camera but the person is actually looking at it due to the eye movements? At first tries, we studied the problem of gaze detection trying to get the direction of the eye using a sort of perspective and point estimation. 

This is actually really hard to get due to several causes. The main problem is that the majority of gaze estimation cited in the literature makes use of a calibration procedure that measures pupil movement while the eyes are looking at dots on a screen. This is not feasible with our pipeline, in which the detection happens "in the wild" captured from a moving camera.

To solve the problem we simply considered the pupil position (found using two different methods presented below) with respect to the eye corners. 

The horizontal pupil ratio expresses how the pupil position within the eye, from -0.5 to 0.5, where 0.0 represent the position when the pupil is centered in the eye and 0.5 when the pupil is completely shifted towards the left corner. 

![[pupil_ratio.png]]
The horizontal pupil ratio $hr$ can be found having the half length of the eye $e$ and the distance of the pupil from the eye corner $p$. $hr=(p / e) - 0.5$.

Having the ratio we can compute:
- if the eye is gazing toward the same direction of the face or, 
- if the person is gazing to the camera when the face is facing somewhere else.

As you probably have imagined this method is not suitable for an accurate estimation of the second point, since the direction of the eye is really biased from the distance between the person and the camera. The first point, on the other hand, can be solved quickly just setting a threshold as we have done in with the face facing.

Regarding the second point, we made a proof of concept setting some parameters that are corrected with a person standing around 1m far from the camera. 

### Means of Gradient

To get the precise pupil localisation we have read in detail the method proposed by [Timm and Barth, 2011]. This method lets us get the exact center of the pupil also in images with low resolution and bad lighting. The searched point can be found by comparing the gradient vector $g_i$ at position $x_i$ with the displacement vector of a possible center $d_i$. 



$$  c^*=\underset{c}{\arg\min} \frac{1}{N} \sum_{i=1}^{N}w_c(d_i^\top g_i)^2 , $$ 

$$ d_i = \frac{x_i-c}{||x_i-c||_2}, \forall{i}: ||g_i||_2=1 $$

This method requires an input of a cropped image of an eye, so we used the information from the second step to get the needed processing. The authors of the method suggest also a preprocessing and post-processing phase that have been done to obtain optimal results. Preprocessing is made with the weight $w_c$ helps finding the dark pupils, since it is the grey value at $(c_x, c_y)$ of the smoothed and inverted input image. A gaussian filter also is needed to remove bright outliers like reflections of the glasses. Then as post-processing a threshold has been applied to remove possible results connected to borders, like eyebrows, glasses or hair. 

We then developed a script of the above-presented method, partially following an already existing work [Trishume].

### Filtering

The proposed method is very interesting and works very well in difficult condition, but it is also very computational intense. In real-time systems may not be feasible. So, we tried and implemented a simple but effective filtering approach that works very well and requires less effort. 
Having the cropped eye, we apply a series of filtering and preprocessing aimed to make stand out the pupil, that it is always the darker and more circular object in eye.
In particular, we increased the contrast of the image and we equalised it. Then we applied a soft gaussian blur to remove the noise and a erosion filter that removes the smaller parts like the eyelashes. We then use the OpenCv adaptive thresholding method to binarize the image and lastly we get the counters of all of the blob. 
Looking for the largest one we find the pupil. See the figure for the application of the approach. 


![Example of the above-reported results](face1_edited.png) 
*Figure 1. Results of the first part of the pipeline.**

### Testing

To compare the two method we run both of them over a dataset containing around 1500 images of faces, all hand labeled with the exact position of the pupils.
We measures the just found pupils positions with comparing them with the labels, computing the Hausdorff distance (described in [Huttenlocher et all]) that is the maximum all the distances between each point of a set and the closest point in the other set. For each image of the dataset we found the distances and then computed the mean.$$H(A, B) = \max(h(A,B), h(B,A))$$ $$h(A,B) = \max_{a\in A}\min_{b \in B} ||a-b||$$
The results are down below. Please note that the mean and the standard deviation are done after removing outliers that where completely wrong (Z-score greater than 5). It is pretty straightforward to notice how the average time to compute the position is about 4 time longer with the means of gradient then with filtering, the means are quite the same and despite the fact that the first method is slightly more precise, it has also an higher deviation.


| |Mean|Std. Dev.| Outliers | Avg. time |
|-|-|-|-|-|
| Means of Gradient |3.004 | 2.220 | 3 | 0.112
| Filtering | 3.198|1.156| 3 | 0.028


It has not been said that the first method is actually much more precise in low light environment, but for situation like ours real-time detection in normal environment, we have chose to use the filtering method. 

## Camera tracking



# References

- Timm, F., & Barth, E. (2011). Accurate eye centre localisation by means of gradients. Visapp, 11, 125-130
- Vahid Kazemi, Josephine Sullivan; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 1867-1874
- Viola, P., & Jones, M. J. (2004). Robust real-time face detection. International journal of computer vision, 57, 137-154.
- abhisuri97. (n.d.). Abhisuri97/pyeLike: Basic pupil tracking with gradients in Python (based on Fabian Timm's algorithm). GitHub. Retrieved May 1, 2023, from https://github.com/abhisuri97/pyeLike 
- Trishume. (n.d.). Trishume/eyeLike: A webcam based pupil tracking implementation. GitHub. Retrieved May 1, 2023, from https://github.com/trishume/eyeLike 
- Zhao, W., Chellappa, R., & Phillips, P. J. (2003). A. Rosenfeld. Face recognition: a literature survey. ACM Computing Surveys, 35(4), 399-458.
- https://www.bioid.com/uploads/AVBPA01BioID.pdf
- IBUG 300-W Dataset
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=232073&casa_token=CiQ3K0lLncQAAAAA:Il1pj7-8q7zMGKOhCnsa4GWAc7p-cfxc6oDzT0QqqorPWQmzQcDFn2e0Thm_rQExtLmjzlm57g
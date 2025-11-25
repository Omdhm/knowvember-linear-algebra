from manimlib import *
import numpy as np
import os

# ---------- HELPER FUNCTIONS ----------
def np2img_file(arr, path):
    """Save a grayscale numpy array to PNG and return path."""
    try:
        from PIL import Image
        # Normalize to 0-255
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        arr_clip = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr_clip).save(path)
        return path
    except Exception as e:
        print(f"Error saving image {path}: {e}")
        return None

class LinearAlgebraCourse(Scene):
    def construct(self):
        # Setup
        self.setup_data()
        
        # The Narrative Arc (The "Course")
        self.chapter_1_vectors()
        self.chapter_2_linear_combinations()
        self.chapter_3_linear_transformations()
        self.chapter_4_eigenvectors()
        self.chapter_5_eigenfaces()
        
        # Outro
        self.outro()

    def setup_data(self):
        """Pre-load data to avoid lag during animation"""
        try:
            from sklearn.datasets import fetch_lfw_people
            from sklearn.decomposition import PCA
            
            data_home = os.path.join(os.getcwd(), "sklearn_data")
            faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                                     color=False, data_home=data_home, download_if_missing=True)
            self.faces = faces
            self.h, self.w = faces.images.shape[1], faces.images.shape[2]
            
            # Compute PCA
            self.pca = PCA(n_components=20, svd_solver="randomized", whiten=True).fit(faces.data)
            
        except Exception as e:
            print(f"Data setup failed: {e}")
            self.faces = None

    def title_card(self, title_text, subtitle_text=""):
        title = Text(title_text, font_size=60, color=BLUE)
        subtitle = Text(subtitle_text, font_size=32, color=GREY).next_to(title, DOWN)
        self.play(FadeIn(title, shift=UP), FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

    # ==========================================
    # CHAPTER 1: VECTORS
    # ==========================================
    def chapter_1_vectors(self):
        self.title_card("Chapter 1: Vectors", "The Building Blocks")
        
        # 1. The Grid
        plane = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))
        plane.set_opacity(0.5)
        self.play(ShowCreation(plane))
        
        # 2. What is a vector?
        # "To a physicist, it's an arrow. To a CS student, it's a list."
        
        v = Vector([2, 1], color=YELLOW)
        label = Tex(r"\vec{v} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}", color=YELLOW).next_to(v.get_end(), UR)
        
        self.play(GrowArrow(v))
        self.play(Write(label))
        self.wait(1)
        
        # 3. Operations: Scaling and Addition
        # Scaling
        v_scaled = Vector([4, 2], color=ORANGE)
        label_scaled = Tex(r"2 \cdot \vec{v} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}", color=ORANGE).next_to(v_scaled.get_end(), UR)
        
        self.play(Transform(v, v_scaled), Transform(label, label_scaled))
        self.wait(1)
        
        # Reset
        self.play(FadeOut(v), FadeOut(label))
        
        # Addition
        v1 = Vector([1, 2], color=BLUE)
        v2 = Vector([2, -1], color=RED).shift(v1.get_end()) # Tip-to-tail
        v_sum = Vector([3, 1], color=PURPLE)
        
        label_sum = Tex(r"\vec{v}_1 + \vec{v}_2", color=PURPLE).next_to(v_sum.get_end(), RIGHT)
        
        self.play(GrowArrow(v1))
        self.play(GrowArrow(v2))
        self.play(GrowArrow(v_sum), Write(label_sum))
        self.wait(2)
        
        self.play(FadeOut(Group(plane, v1, v2, v_sum, label_sum)))

    # ==========================================
    # CHAPTER 2: LINEAR COMBINATIONS
    # ==========================================
    def chapter_2_linear_combinations(self):
        self.title_card("Chapter 2: Linear Combinations", "Spanning the Space")
        
        plane = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))
        plane.set_opacity(0.3)
        self.play(FadeIn(plane))
        
        # 1. Basis Vectors i-hat and j-hat
        i_hat = Vector([1, 0], color=GREEN)
        j_hat = Vector([0, 1], color=RED)
        
        label_i = Tex(r"\hat{i}", color=GREEN).next_to(i_hat.get_end(), DOWN)
        label_j = Tex(r"\hat{j}", color=RED).next_to(j_hat.get_end(), LEFT)
        
        self.play(GrowArrow(i_hat), Write(label_i))
        self.play(GrowArrow(j_hat), Write(label_j))
        
        # 2. Linear Combination
        # v = 3i - 2j
        
        formula = Tex(r"\vec{v} = 3\hat{i} - 2\hat{j}", font_size=48).to_edge(TOP)
        self.play(Write(formula))
        
        # Animate the components
        c1 = Vector([3, 0], color=GREEN).set_opacity(0.7)
        c2 = Vector([0, -2], color=RED).set_opacity(0.7).shift([3, 0, 0])
        v_res = Vector([3, -2], color=YELLOW)
        
        self.play(TransformFromCopy(i_hat, c1))
        self.play(TransformFromCopy(j_hat, c2))
        self.play(GrowArrow(v_res))
        self.wait(2)
        
        self.play(FadeOut(Group(plane, i_hat, j_hat, label_i, label_j, formula, c1, c2, v_res)))

    # ==========================================
    # CHAPTER 3: LINEAR TRANSFORMATIONS
    # ==========================================
    def chapter_3_linear_transformations(self):
        self.title_card("Chapter 3: Linear Transformations", "Matrices as Functions")
        
        # 1. Setup Grid
        grid = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))
        grid.prepare_for_nonlinear_transform()
        self.play(ShowCreation(grid))
        
        # 2. Show Basis Vectors
        i_hat = Vector([1, 0], color=GREEN)
        j_hat = Vector([0, 1], color=RED)
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        
        # 3. The Matrix
        # A = [[1, 1], [0, 1]] (Shear)
        matrix_tex = Tex(r"A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}", color=BLUE).to_corner(UL)
        self.play(Write(matrix_tex))
        
        # 4. Apply Transformation
        # i_hat stays at [1, 0]
        # j_hat moves to [1, 1]
        
        matrix = [[1, 1], [0, 1]]
        
        self.play(
            ApplyMatrix(matrix, grid),
            ApplyMatrix(matrix, i_hat),
            ApplyMatrix(matrix, j_hat),
            run_time=3
        )
        self.wait(1)
        
        # Show where j_hat landed
        label_new_j = Tex(r"\text{New } \hat{j} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}", color=RED).next_to(j_hat.get_end(), UR)
        self.play(Write(label_new_j))
        self.wait(2)
        
        self.play(FadeOut(Group(grid, i_hat, j_hat, matrix_tex, label_new_j)))

    # ==========================================
    # CHAPTER 4: EIGENVECTORS
    # ==========================================
    def chapter_4_eigenvectors(self):
        self.title_card("Chapter 4: Eigenvectors", "The Axes of Rotation")
        
        # 1. Setup
        grid = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))
        self.play(FadeIn(grid))
        
        # 2. Symmetric Matrix (Scaling along axes)
        # A = [[2, 0], [0, 0.5]]
        matrix_tex = Tex(r"A = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}", color=PURPLE).to_corner(UL)
        self.play(Write(matrix_tex))
        
        # 3. Show vectors
        v1 = Vector([1, 1], color=GREY) # Will change direction
        v_eig = Vector([1, 0], color=YELLOW) # Will NOT change direction
        
        self.play(GrowArrow(v1), GrowArrow(v_eig))
        
        # 4. Transform
        matrix = [[2, 0], [0, 0.5]]
        
        # v1 -> [2, 0.5]
        # v_eig -> [2, 0]
        
        v1_target = Vector([2, 0.5], color=GREY).set_opacity(0.5)
        v_eig_target = Vector([2, 0], color=YELLOW)
        
        self.play(
            TransformFromCopy(v1, v1_target),
            TransformFromCopy(v_eig, v_eig_target),
            run_time=2
        )
        
        # 5. Analyze
        cross = Cross(v1_target).scale(0.5)
        check = Tex(r"\checkmark", color=GREEN).next_to(v_eig_target, UP)
        
        self.play(ShowCreation(cross))
        self.play(Write(check))
        
        eigen_def = Tex(r"A \vec{v} = \lambda \vec{v}", color=YELLOW).next_to(matrix_tex, DOWN)
        self.play(Write(eigen_def))
        
        txt = Text("Eigenvectors stay on their span!", font_size=32, color=YELLOW).to_edge(BOTTOM)
        self.play(Write(txt))
        self.wait(2)
        
        self.play(FadeOut(Group(grid, matrix_tex, v1, v_eig, v1_target, v_eig_target, cross, check, eigen_def, txt)))

    # ==========================================
    # CHAPTER 5: APPLICATION (EIGENFACES)
    # ==========================================
    def chapter_5_eigenfaces(self):
        self.title_card("Chapter 5: Application", "Eigenfaces")
        
        if self.faces is None:
            err = Text("Dataset not loaded", color=RED)
            self.play(Write(err))
            return

        # 1. The "Face Space"
        # "An image is a vector in 1850D space"
        
        face_arr = self.faces.images[0]
        face_path = np2img_file(face_arr, "lfw_chap5.png")
        face_img = ImageMobject(face_path).scale(2).to_edge(LEFT, buff=2)
        
        self.play(FadeIn(face_img))
        
        arrow = Arrow(face_img.get_right(), RIGHT*2, color=GREY)
        vec_txt = Tex(r"\vec{x} \in \mathbb{R}^{1850}", font_size=48).next_to(arrow, RIGHT)
        
        self.play(GrowArrow(arrow), Write(vec_txt))
        self.wait(1)
        self.play(FadeOut(arrow), FadeOut(vec_txt))
        
        # 2. The Basis Vectors (Eigenfaces)
        # "We found the eigenvectors of the face covariance matrix"
        
        h, w = self.h, self.w
        
        # Mean Face
        mean_img = self.pca.mean_.reshape(h, w)
        mean_path = np2img_file(mean_img, "mean_face.png")
        mean_mob = ImageMobject(mean_path).scale(1.5)
        mean_label = Tex(r"\vec{\mu} \text{ (Mean)}", font_size=24, color=BLUE).next_to(mean_mob, DOWN)
        
        mean_group = Group(mean_mob, mean_label).to_edge(LEFT, buff=1)
        
        self.play(Transform(face_img, mean_mob), Write(mean_label)) # Transform original to mean as starting point
        
        # Eigenfaces
        eig_mobs = Group()
        for i in range(3):
            comp = self.pca.components_[i].reshape(h, w)
            path = np2img_file(comp, f"eig_{i}.png")
            mob = ImageMobject(path).scale(1.2)
            eig_mobs.add(mob)
            
        eig_mobs.arrange(RIGHT, buff=0.3).next_to(mean_group, RIGHT, buff=1)
        eig_label = Text("Eigenvectors (Basis)", font_size=24, color=PURPLE).next_to(eig_mobs, UP)
        
        self.play(FadeIn(eig_mobs), Write(eig_label))
        self.wait(1)
        
        # 3. Reconstruction (Linear Combination)
        # "Any face is a linear combination of these eigenvectors"
        
        formula = Tex(r"\vec{x} \approx \vec{\mu} + w_1 \vec{v}_1 + w_2 \vec{v}_2 + \dots", font_size=32, color=YELLOW)
        formula.to_edge(BOTTOM)
        self.play(Write(formula))
        
        # Animate adding them up
        # We'll just show the result of adding top 10
        
        target_vec = self.faces.data[0]
        weights = self.pca.transform([target_vec])[0]
        
        # Reconstruct with 10 components
        rec_vec = self.pca.mean_ + np.dot(weights[:10], self.pca.components_[:10])
        rec_img = rec_vec.reshape(h, w)
        rec_path = np2img_file(rec_img, "rec_final.png")
        
        rec_mob = ImageMobject(rec_path).scale(1.5).next_to(eig_mobs, RIGHT, buff=1)
        rec_lbl = Text("Result", font_size=24, color=GREEN).next_to(rec_mob, DOWN)
        
        plus = Tex("+", font_size=48).next_to(mean_group, RIGHT, buff=0.2)
        equals = Tex("=", font_size=48).next_to(eig_mobs, RIGHT, buff=0.2)
        
        self.play(Write(plus), Write(equals))
        self.play(FadeIn(rec_mob), Write(rec_lbl))
        self.wait(3)
        
        self.play(FadeOut(Group(mean_group, eig_mobs, eig_label, rec_mob, rec_lbl, plus, equals, formula)))

    def outro(self):
        t = Text("Linear Algebra gives us the eyes to see high-dimensional space.", font_size=32, color=BLUE)
        self.play(Write(t))
        self.wait(3)
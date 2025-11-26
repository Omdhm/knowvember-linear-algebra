from manimlib import *
import numpy as np
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

class Colors:
    PRIMARY = "#3B82F6"
    SECONDARY = "#8B5CF6"
    ACCENT = "#F59E0B"
    SUCCESS = "#10B981"
    ERROR = "#EF4444"
    I_HAT = "#22C55E"
    J_HAT = "#EF4444"
    RESULT = "#FBBF24"
    EIGEN_1 = "#06B6D4"
    EIGEN_2 = "#F472B6"
    GRID_MAIN = "#64748B"
    GRID_FADED = "#334155"
    TEXT_PRIMARY = "#F8FAFC"
    TEXT_SECONDARY = "#94A3B8"


class Config:
    TITLE_FONT_SIZE = 72
    SUBTITLE_FONT_SIZE = 36
    BODY_FONT_SIZE = 32
    LABEL_FONT_SIZE = 28
    TITLE_DURATION = 2.5
    TRANSITION_DURATION = 0.8


# =============================================================================
# HELPER
# =============================================================================

def save_array_as_image(arr: np.ndarray, path: str) -> str:
    try:
        from PIL import Image
        arr_normalized = arr.astype(np.float64)
        if arr_normalized.max() > arr_normalized.min():
            arr_normalized = (arr_normalized - arr_normalized.min())
            arr_normalized /= (arr_normalized.max() - arr_normalized.min())
            arr_normalized *= 255
        arr_uint8 = np.clip(arr_normalized, 0, 255).astype(np.uint8)
        Image.fromarray(arr_uint8, mode='L').save(path)
        return path
    except Exception as e:
        print(f"[ERROR] Failed to save image {path}: {e}")
        return None


# =============================================================================
# MAIN CLASS
# =============================================================================

class LinearAlgebraCourse(Scene):
    
    def construct(self):
        self._setup_resources()
        
        self._intro_sequence()
        self._chapter_vectors()
        self._chapter_linear_combinations()
        self._chapter_transformations()
        self._chapter_determinants()
        self._chapter_eigenvectors()
        self._chapter_pca_eigenfaces()
        self._outro_sequence()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    def _setup_resources(self):
        self.image_dir = Path("./generated_images")
        self.image_dir.mkdir(exist_ok=True)
        self._load_face_data()
    
    def _load_face_data(self):
        try:
            from sklearn.datasets import fetch_lfw_people
            from sklearn.decomposition import PCA
            
            print("[INFO] Loading LFW face dataset...")
            data_home = Path("./sklearn_data")
            
            self.faces = fetch_lfw_people(
                min_faces_per_person=70,
                resize=0.4,
                color=False,
                data_home=str(data_home),
                download_if_missing=True
            )
            
            self.face_height = self.faces.images.shape[1]
            self.face_width = self.faces.images.shape[2]
            
            print("[INFO] Computing PCA...")
            self.pca_model = PCA(
                n_components=150,
                svd_solver="randomized",
                whiten=True,
                random_state=42
            ).fit(self.faces.data)
            
            print("[INFO] Done!")
            
        except Exception as e:
            print(f"[WARNING] Could not load face data: {e}")
            self.faces = None
            self.pca_model = None
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _create_title_card(self, title: str, subtitle: str = "", duration: float = None):
        duration = duration or Config.TITLE_DURATION
        
        title_mob = Text(title, font_size=Config.TITLE_FONT_SIZE, color=Colors.PRIMARY, weight=BOLD)
        title_glow = title_mob.copy()
        title_glow.set_stroke(Colors.PRIMARY, width=8, opacity=0.3)
        title_group = VGroup(title_glow, title_mob)
        
        elements = [title_group]
        if subtitle:
            subtitle_mob = Text(subtitle, font_size=Config.SUBTITLE_FONT_SIZE, color=Colors.TEXT_SECONDARY)
            subtitle_mob.next_to(title_mob, DOWN, buff=0.4)
            elements.append(subtitle_mob)
        
        group = VGroup(*elements)
        
        self.play(FadeIn(title_group, shift=UP * 0.3, scale=0.95), run_time=0.6)
        if subtitle:
            self.play(FadeIn(elements[1], shift=UP * 0.2), run_time=0.4)
        
        self.wait(duration)
        self.play(FadeOut(group, shift=UP * 0.5), run_time=Config.TRANSITION_DURATION)
    
    def _create_chapter_header(self, chapter_num: int, title: str):
        badge = VGroup(
            RoundedRectangle(width=1.8, height=0.7, corner_radius=0.15,
                           fill_color=Colors.PRIMARY, fill_opacity=1, stroke_width=0),
            Text(f"Ch. {chapter_num}", font_size=24, color=WHITE)
        )
        
        title_mob = Text(title, font_size=Config.TITLE_FONT_SIZE - 12, color=Colors.TEXT_PRIMARY)
        badge.next_to(title_mob, UP, buff=0.3)
        group = VGroup(badge, title_mob)
        
        self.play(FadeIn(badge, scale=0.8), FadeIn(title_mob, shift=UP * 0.2), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(group), run_time=0.5)
    
    def _create_grid(self, opacity: float = 0.4, color=None):
        color = color or Colors.GRID_MAIN
        return NumberPlane(
            x_range=(-7, 7, 1), y_range=(-5, 5, 1),
            background_line_style={"stroke_color": color, "stroke_width": 1, "stroke_opacity": opacity * 0.5},
            axis_config={"stroke_color": color, "stroke_width": 2, "stroke_opacity": opacity}
        )
    
    # =========================================================================
    # INTRO
    # =========================================================================
    
    def _intro_sequence(self):
        # 3D Intro
        title = Text("Linear Algebra", font_size=Config.TITLE_FONT_SIZE, color=Colors.PRIMARY, weight=BOLD)
        subtitle = Text("The Language of Multidimensional Space", font_size=Config.SUBTITLE_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        subtitle.next_to(title, DOWN, buff=0.5)
        
        group = VGroup(title, subtitle)
        
        # Start with camera rotation
        frame = self.camera.frame
        frame.set_euler_angles(theta=-30 * DEGREES, phi=70 * DEGREES)
        
        self.play(DrawBorderThenFill(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP))
        
        # Rotate camera back to normal
        self.play(
            frame.animate.set_euler_angles(theta=0, phi=0),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1)
        self.play(FadeOut(group, shift=UP))
        
        self._show_applications()
    
    def _show_applications(self):
        applications = [
            ("Robotics", "Motion planning"),
            ("Graphics", "3D rendering"),
            ("ML", "Data analysis"),
            ("Vision", "Image recognition")
        ]
        
        cards = VGroup()
        for name, desc in applications:
            card = VGroup(
                RoundedRectangle(width=2.8, height=1.2, corner_radius=0.1,
                               fill_color=Colors.GRID_FADED, fill_opacity=0.8,
                               stroke_color=Colors.PRIMARY, stroke_width=2),
                Text(name, font_size=24, color=Colors.PRIMARY, weight=BOLD),
                Text(desc, font_size=14, color=Colors.TEXT_SECONDARY)
            )
            card[1].move_to(card[0].get_center() + UP * 0.2)
            card[2].move_to(card[0].get_center() + DOWN * 0.2)
            cards.add(card)
        
        cards.arrange_in_grid(n_rows=2, n_cols=2, buff=0.3)
        
        # Pop in effect
        self.play(
            LaggedStart(*[
                AnimationGroup(
                    FadeIn(c, scale=0.5),
                    Flash(c, color=Colors.PRIMARY, line_length=0.2, num_lines=4, flash_radius=1.5)
                ) for c in cards
            ], lag_ratio=0.3),
            run_time=2.5
        )
        self.wait(2)
        self.play(FadeOut(cards), run_time=0.8)
    
    # =========================================================================
    # CHAPTER 1: VECTORS
    # =========================================================================
    
    def _chapter_vectors(self):
        self._create_chapter_header(1, "Vectors")
        
        plane = self._create_grid()
        self.play(ShowCreation(plane, lag_ratio=0.1), run_time=2)
        
        # What is a vector
        explanation = Text("A vector is an arrow with direction and magnitude",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY).to_edge(UP, buff=0.5)
        self.play(Write(explanation))
        
        v = Vector([3, 2], color=Colors.RESULT, stroke_width=5)
        self.play(GrowArrow(v))
        
        # Wiggle to emphasize "direction and magnitude"
        self.play(WiggleOutThenIn(v))
        
        coords = Tex(r"\vec{v} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}",
                    font_size=40, color=Colors.RESULT).next_to(v.get_end(), UR, buff=0.2)
        self.play(Write(coords))
        self.wait(1.5)
        
        # Scaling
        self.play(FadeOut(explanation))
        scale_text = Text("Scaling: multiply each component",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY).to_edge(UP, buff=0.5)
        self.play(Write(scale_text))
        
        v_scaled = Vector([4.5, 3], color=Colors.ACCENT, stroke_width=5)
        coords_scaled = Tex(r"1.5 \cdot \vec{v} = \begin{bmatrix} 4.5 \\ 3 \end{bmatrix}",
                           font_size=40, color=Colors.ACCENT).next_to(v_scaled.get_end(), UR, buff=0.2)
        
        # Flash effect on scaling
        self.play(
            Transform(v, v_scaled),
            Transform(coords, coords_scaled),
            Flash(v_scaled.get_end(), color=Colors.ACCENT),
            run_time=1.5
        )
        self.wait(1.5)
        self.play(FadeOut(v), FadeOut(coords), FadeOut(scale_text))
        
        # Addition
        add_text = Text("Addition: tip-to-tail",
                       font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY).to_edge(UP, buff=0.5)
        self.play(Write(add_text))
        
        v1 = Vector([2, 3], color=Colors.PRIMARY, stroke_width=4)
        v2 = Vector([3, -1], color=Colors.SECONDARY, stroke_width=4).shift(v1.get_end())
        
        label_v1 = Tex(r"\vec{a}", color=Colors.PRIMARY, font_size=32).next_to(v1, LEFT, buff=0.1)
        label_v2 = Tex(r"\vec{b}", color=Colors.SECONDARY, font_size=32).next_to(v2, UP, buff=0.1)
        
        self.play(GrowArrow(v1), Write(label_v1))
        self.play(GrowArrow(v2), Write(label_v2))
        
        v_sum = Vector([5, 2], color=Colors.SUCCESS, stroke_width=5)
        label_sum = Tex(r"\vec{a} + \vec{b}", color=Colors.SUCCESS, font_size=32).next_to(v_sum.get_end(), RIGHT, buff=0.1)
        
        self.play(GrowArrow(v_sum), Write(label_sum))
        self.play(Indicate(v_sum, color=Colors.SUCCESS)) # Pulse the result
        self.wait(2)
        
        self.play(FadeOut(VGroup(plane, v1, v2, v_sum, label_v1, label_v2, label_sum, add_text)))

    # ... (rest of file)

    def _pca_act1_problem(self):
        """Act 1: Present the problem - images are huge!"""
        h, w = self.face_height, self.face_width
        n_pixels = h * w
        
        # Show a single face, BIG and CENTERED
        face_array = self.faces.images[0]
        face_path = save_array_as_image(face_array, str(self.image_dir / "face_problem.png"))
        
        face_img = ImageMobject(face_path).scale(3.5)  # BIG
        face_img.move_to(ORIGIN)  # CENTERED
        
        self.play(FadeIn(face_img, scale=0.8), run_time=1)
        
        # Scanning effect
        scan_line = Line(face_img.get_corner(UL), face_img.get_corner(UR), color=Colors.ACCENT, stroke_width=4)
        scan_line.set_width(face_img.get_width())
        
        self.play(ShowCreation(scan_line))
        self.play(
            scan_line.animate.move_to(face_img.get_bottom()),
            run_time=2,
            rate_func=linear
        )
        self.play(FadeOut(scan_line))
        
        # Text BELOW
        problem_text = Text(f"This face = {n_pixels} numbers", font_size=36, color=Colors.TEXT_PRIMARY)
        problem_text.next_to(face_img, DOWN, buff=0.5)
        self.play(Write(problem_text))
        self.wait(1)
        
        # Emphasize the problem
        problem_text2 = Text("That's a LOT of data!", font_size=32, color=Colors.ERROR)
        problem_text2.next_to(problem_text, DOWN, buff=0.3)
        self.play(Write(problem_text2))
        self.play(WiggleOutThenIn(problem_text2))
        self.wait(2)
        
        # Clear
        self.play(FadeOut(Group(face_img, problem_text, problem_text2)))
        
        # Question
        question = Text("Can we use fewer numbers?", font_size=48, color=Colors.ACCENT)
        question.move_to(ORIGIN)
        self.play(Write(question))
        self.wait(2)
        self.play(FadeOut(question))
    
    # =========================================================================
    # CHAPTER 2: LINEAR COMBINATIONS
    # =========================================================================
    
    def _chapter_linear_combinations(self):
        self._create_chapter_header(2, "Linear Combinations")
        
        plane = self._create_grid()
        self.play(FadeIn(plane), run_time=0.8)
        
        i_hat = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        j_hat = Vector([0, 1], color=Colors.J_HAT, stroke_width=5)
        
        i_label = Tex(r"\hat{\imath}", color=Colors.I_HAT, font_size=36).next_to(i_hat.get_end(), DOWN, buff=0.1)
        j_label = Tex(r"\hat{\jmath}", color=Colors.J_HAT, font_size=36).next_to(j_hat.get_end(), LEFT, buff=0.1)
        
        self.play(GrowArrow(i_hat), Write(i_label), GrowArrow(j_hat), Write(j_label))
        
        formula = Tex(r"\vec{v} = a\hat{\imath} + b\hat{\jmath}", font_size=44).to_edge(UP, buff=0.6)
        self.play(Write(formula))
        
        example = Tex(r"\vec{v} = 3\hat{\imath} - 2\hat{\jmath}", font_size=44).next_to(formula, DOWN, buff=0.3)
        self.play(Write(example))
        
        comp_i = Vector([3, 0], color=Colors.I_HAT, stroke_width=3).set_opacity(0.6)
        comp_j = Vector([0, -2], color=Colors.J_HAT, stroke_width=3).set_opacity(0.6).shift([3, 0, 0])
        
        result = Vector([3, -2], color=Colors.RESULT, stroke_width=5)
        result_label = Tex(r"\vec{v}", color=Colors.RESULT, font_size=36).next_to(result.get_end(), DR, buff=0.1)
        
        self.play(TransformFromCopy(i_hat, comp_i))
        self.play(TransformFromCopy(j_hat, comp_j))
        self.play(GrowArrow(result), Write(result_label))
        self.wait(2)
        
        self.play(FadeOut(VGroup(plane, i_hat, j_hat, i_label, j_label, formula, example, comp_i, comp_j, result, result_label)))
    
    # =========================================================================
    # CHAPTER 3: TRANSFORMATIONS
    # =========================================================================
    
    def _chapter_transformations(self):
        self._create_chapter_header(3, "Linear Transformations")
        
        # 1. Setup the Space
        grid = self._create_grid(opacity=0.4)
        grid.prepare_for_nonlinear_transform()
        self.play(ShowCreation(grid, lag_ratio=0.1), run_time=1.5)
        
        # 2. Add "Particles" to visualize the fabric of space
        particles = VGroup()
        for _ in range(50):
            p = np.random.uniform(-4, 4, 3)
            p[2] = 0
            dot = Dot(p, radius=0.05, color=Colors.TEXT_SECONDARY)
            dot.set_opacity(0.6)
            particles.add(dot)
            
        self.play(FadeIn(particles, lag_ratio=0.1))
        
        # 3. Basis Vectors
        i_hat = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        j_hat = Vector([0, 1], color=Colors.J_HAT, stroke_width=5)
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        
        # 4. The Matrix (Rotation + Scale)
        # Rotate 45 degrees and scale x by 1.5
        angle = 45 * DEGREES
        scale_x = 1.5
        matrix_val = [
            [scale_x * np.cos(angle), -np.sin(angle)],
            [scale_x * np.sin(angle), np.cos(angle)]
        ]
        
        matrix_tex = Tex(
            r"A = \begin{bmatrix} 1.06 & -0.71 \\ 1.06 & 0.71 \end{bmatrix}",
            font_size=36, color=Colors.PRIMARY
        ).to_corner(UR, buff=0.5)
        
        label = Text("Rotation + Stretch", font_size=24, color=Colors.ACCENT).next_to(matrix_tex, DOWN)
        
        self.play(Write(matrix_tex), Write(label))
        self.wait(1)
        
        # 5. The Transformation with Tracers
        # Add tracers to basis vectors
        i_trace = TracedPath(i_hat.get_end, stroke_color=Colors.I_HAT, stroke_width=3)
        j_trace = TracedPath(j_hat.get_end, stroke_color=Colors.J_HAT, stroke_width=3)
        self.add(i_trace, j_trace)
        
        self.play(
            ApplyMatrix(matrix_val, grid),
            ApplyMatrix(matrix_val, i_hat),
            ApplyMatrix(matrix_val, j_hat),
            ApplyMatrix(matrix_val, particles), # Move particles too!
            run_time=4,
            rate_func=smooth
        )
        
        # Impact!
        self.play(Flash(i_hat.get_end(), color=Colors.I_HAT), Flash(j_hat.get_end(), color=Colors.J_HAT))
        
        # 6. Insight
        insight = Text("The grid lines remain parallel and evenly spaced", 
                      font_size=24, color=Colors.TEXT_SECONDARY).to_edge(DOWN, buff=0.8)
        self.play(Write(insight))
        self.wait(2)
        
        # Freeze tracers before fading out to avoid shape mismatch errors
        i_trace.clear_updaters()
        j_trace.clear_updaters()
        
        self.play(FadeOut(VGroup(grid, particles, i_hat, j_hat, i_trace, j_trace, matrix_tex, label, insight)))
    
    # =========================================================================
    # CHAPTER 4: DETERMINANTS
    # =========================================================================
    
    def _chapter_determinants(self):
        self._create_chapter_header(4, "Determinants")
        
        intro = Text("The determinant measures how area scales",
                    font_size=28, color=Colors.TEXT_SECONDARY).to_edge(UP, buff=0.5)
        self.play(Write(intro))
        self.wait(1)
        self.play(FadeOut(intro))
        
        # Example 1: Scaling
        self._det_example("Scaling", [[3, 0], [0, 2]], r"\det = 3 \times 2 = 6", Colors.RESULT)
        
        # Example 2: Shear
        self._det_example("Shear (det=1)", [[1, 1], [0, 1]], r"\det = 1 \times 1 - 1 \times 0 = 1", Colors.ACCENT)
        
        # Example 3: Singular
        self._det_example("Singular (det=0)", [[1, 2], [0.5, 1]], r"\det = 1 - 1 = 0", Colors.ERROR)
    
    def _det_example(self, title_text, matrix, det_text, color):
        title = Text(title_text, font_size=32, color=Colors.PRIMARY).to_edge(UP, buff=0.4)
        self.play(Write(title))
        
        plane = self._create_grid(opacity=0.3)
        self.play(FadeIn(plane), run_time=0.5)
        
        square = Square(side_length=1, color=color, fill_opacity=0.5, stroke_width=3)
        square.move_to([0.5, 0.5, 0])
        self.play(ShowCreation(square))
        
        self.play(ApplyMatrix(matrix, plane), ApplyMatrix(matrix, square), run_time=2)
        
        det_formula = Tex(det_text, font_size=32, color=color).to_edge(DOWN, buff=0.6)
        self.play(Write(det_formula))
        self.wait(1.5)
        
        self.play(FadeOut(VGroup(title, plane, square, det_formula)))
    
    # =========================================================================
    # CHAPTER 5: EIGENVECTORS
    # =========================================================================
    
    def _chapter_eigenvectors(self):
        self._create_chapter_header(5, "Eigenvectors")
        
        # Introduction
        question = Text("Which vectors only get stretched, not rotated?",
                       font_size=30, color=Colors.TEXT_SECONDARY).to_edge(UP, buff=0.8)
        self.play(Write(question))
        
        eigen_eq = Tex(r"A\vec{v} = \lambda\vec{v}", font_size=56, color=Colors.RESULT)
        self.play(Write(eigen_eq))
        
        explanations = VGroup(
            Tex(r"\vec{v}", font_size=36, color=Colors.SUCCESS),
            Text(" = eigenvector (special direction)", font_size=24, color=Colors.TEXT_SECONDARY),
        ).arrange(RIGHT, buff=0.1)
        explanations2 = VGroup(
            Tex(r"\lambda", font_size=36, color=Colors.ACCENT),
            Text(" = eigenvalue (scaling factor)", font_size=24, color=Colors.TEXT_SECONDARY),
        ).arrange(RIGHT, buff=0.1)
        
        all_exp = VGroup(explanations, explanations2).arrange(DOWN, buff=0.3).next_to(eigen_eq, DOWN, buff=0.6)
        self.play(Write(all_exp))
        self.wait(2)
        self.play(FadeOut(VGroup(question, eigen_eq, all_exp)))
        
        # Demonstration
        self._eigen_demo()
    
    def _eigen_demo(self):
        title = Text("Eigenvectors stay on their line!", font_size=28, color=Colors.TEXT_SECONDARY).to_edge(UP, buff=0.4)
        self.play(Write(title))
        
        grid = self._create_grid(opacity=0.3)
        grid.prepare_for_nonlinear_transform()
        self.play(FadeIn(grid))
        
        matrix_tex = Tex(r"A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}",
                        font_size=36, color=Colors.PRIMARY).to_corner(UL, buff=0.5)
        self.play(Write(matrix_tex))
        
        # Eigenvector [1,0] and non-eigenvector [1,1]
        v_eigen = Vector([2, 0], color=Colors.SUCCESS, stroke_width=5)
        v_other = Vector([1, 1], color=Colors.ERROR, stroke_width=5)
        
        eigen_label = Text("Eigenvector", font_size=18, color=Colors.SUCCESS).next_to(v_eigen.get_end(), UP, buff=0.1)
        other_label = Text("Regular vector", font_size=18, color=Colors.ERROR).next_to(v_other.get_end(), UR, buff=0.1)
        
        self.play(GrowArrow(v_eigen), Write(eigen_label), GrowArrow(v_other), Write(other_label))
        
        span_line = DashedLine([-5, 0, 0], [5, 0, 0], color=Colors.SUCCESS, stroke_width=2)
        self.play(ShowCreation(span_line))
        self.wait(1)
        
        transformation_matrix = [[2, 1], [0, 3]]
        self.play(
            ApplyMatrix(transformation_matrix, grid),
            ApplyMatrix(transformation_matrix, v_eigen),
            ApplyMatrix(transformation_matrix, v_other),
            ApplyMatrix(transformation_matrix, span_line),
            FadeOut(eigen_label), FadeOut(other_label),
            run_time=3
        )
        
        check = Tex(r"\checkmark", font_size=40, color=Colors.SUCCESS).next_to(v_eigen.get_end(), UP, buff=0.1)
        cross = Tex(r"\times", font_size=40, color=Colors.ERROR).next_to(v_other.get_end(), UR, buff=0.1)
        self.play(Write(check), Write(cross))
        
        result = Text("Eigenvector stayed on its line, regular vector didn't!",
                     font_size=24, color=Colors.TEXT_SECONDARY).to_edge(DOWN, buff=0.5)
        self.play(Write(result))
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, grid, matrix_tex, v_eigen, v_other, span_line, check, cross, result)))
    
    # =========================================================================
    # CHAPTER 6: PCA & EIGENFACES - COMPLETELY REDESIGNED
    # =========================================================================
    
    def _chapter_pca_eigenfaces(self):
        self._create_chapter_header(6, "PCA & Eigenfaces")
        
        if self.faces is None or self.pca_model is None:
            return
            
        # 1. The Data Cloud (3D Visualization)
        self._pca_scene_cloud()
        
        # 2. The "Recipe" (Equalizer Concept)
        self._pca_scene_equalizer()
        
    def _pca_scene_cloud(self):
        """Visualize faces as points in a high-dimensional cloud."""
        # Title
        title = Text("Imagine every face is a point...", font_size=36, color=Colors.PRIMARY)
        title.to_edge(UP, buff=1)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Create 3D axes
        axes = ThreeDAxes()
        self.play(ShowCreation(axes))
        
        # Create a cloud of points (representing faces)
        # We project the first 3 PCs to 3D space
        points = VGroup()
        pca_3d = self.pca_model.transform(self.faces.data[:100])[:, :3]
        
        # Normalize for display
        pca_3d = pca_3d / np.max(np.abs(pca_3d)) * 3
        
        for p in pca_3d:
            dot = Dot3D(point=p, radius=0.08, color=Colors.ACCENT)
            points.add(dot)
            
        self.play(ShowCreation(points, lag_ratio=0.05), run_time=3)
        
        # Rotate camera around the cloud
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, run_time=2)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        
        # Show the "Best Fit" line (PC1)
        pc1_line = Line3D(start=[0,0,0], end=[4, 0, 0], color=Colors.EIGEN_1, thickness=0.05)
        # Rotate line to align with data variance (simulated)
        self.play(ShowCreation(pc1_line))
        self.play(Rotate(pc1_line, angle=45*DEGREES, axis=UP), run_time=2)
        
        label_pc1 = Text("Principal Component 1", font_size=24, color=Colors.EIGEN_1)
        self.add_fixed_in_frame_mobjects(label_pc1)
        label_pc1.to_corner(DR)
        self.play(Write(label_pc1))
        self.wait(2)
        
        # Transition out
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=0, theta=0, run_time=1.5)
        self.play(
            FadeOut(points), FadeOut(axes), FadeOut(pc1_line), 
            FadeOut(title), FadeOut(label_pc1)
        )
        
    def _pca_scene_equalizer(self):
        """The 'Equalizer' view of face reconstruction."""
        h, w = self.face_height, self.face_width
        
        # Layout: Face on Left, Sliders on Right
        title = Text("The Face Equalizer", font_size=40, color=Colors.PRIMARY)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        
        # Target Face
        target_idx = 15
        target_vec = self.faces.data[target_idx]
        weights = self.pca_model.transform([target_vec])[0]
        
        # Start with Mean Face
        current_vec = self.pca_model.mean_.copy()
        current_img_path = save_array_as_image(current_vec.reshape(h, w), str(self.image_dir / "eq_mean.png"))
        face_mob = ImageMobject(current_img_path).scale(3).to_edge(LEFT, buff=2)
        
        face_label = Text("Mean Face", font_size=24, color=Colors.TEXT_SECONDARY).next_to(face_mob, DOWN)
        
        self.play(FadeIn(face_mob), Write(face_label))
        
        # Create Sliders (Visual representation of weights)
        sliders = VGroup()
        knobs = VGroup()
        labels = VGroup()
        
        start_x = 2
        spacing = 1.5
        
        # Show top 3 components
        for i in range(3):
            # Track line
            line = Line(UP*1.5, DOWN*1.5, color=Colors.GRID_FADED)
            line.move_to([start_x + i*spacing, 0, 0])
            
            # Knob (starts at 0 center)
            knob = Dot(color=Colors.ACCENT, radius=0.15)
            knob.move_to(line.get_center())
            
            # Label
            label = Tex(rf"w_{i+1}", color=Colors.TEXT_PRIMARY).next_to(line, UP)
            
            sliders.add(line)
            knobs.add(knob)
            labels.add(label)
            
        self.play(ShowCreation(sliders), FadeIn(knobs), Write(labels))
        
        # Animate Sliders moving and Face updating
        # Component 1
        w1 = weights[0]
        # Normalize weight for slider visual (just for show)
        y_pos = np.clip(w1 / 1000, -1.5, 1.5) 
        
        self.play(
            knobs[0].animate.move_to(sliders[0].get_center() + UP * y_pos),
            run_time=1.5
        )
        
        # Update Face (Add PC1)
        current_vec += w1 * self.pca_model.components_[0]
        path_1 = save_array_as_image(current_vec.reshape(h, w), str(self.image_dir / "eq_step1.png"))
        face_1 = ImageMobject(path_1).scale(3).move_to(face_mob)
        
        self.play(Transform(face_mob, face_1), face_label.animate.become(
            Text("+ Component 1", font_size=24, color=Colors.ACCENT).next_to(face_mob, DOWN)
        ))
        
        # Component 2
        w2 = weights[1]
        y_pos_2 = np.clip(w2 / 1000, -1.5, 1.5)
        
        self.play(
            knobs[1].animate.move_to(sliders[1].get_center() + UP * y_pos_2),
            run_time=1.5
        )
        
        # Update Face (Add PC2)
        current_vec += w2 * self.pca_model.components_[1]
        path_2 = save_array_as_image(current_vec.reshape(h, w), str(self.image_dir / "eq_step2.png"))
        face_2 = ImageMobject(path_2).scale(3).move_to(face_mob)
        
        self.play(Transform(face_mob, face_2), face_label.animate.become(
            Text("+ Component 2", font_size=24, color=Colors.ACCENT).next_to(face_mob, DOWN)
        ))
        
        # Fast forward
        ff_text = Text("... Adding 150 components ...", font_size=32, color=Colors.SUCCESS)
        ff_text.move_to(sliders.get_center())
        
        self.play(FadeOut(sliders), FadeOut(knobs), FadeOut(labels), FadeIn(ff_text))
        
        # Final Result
        final_vec = self.pca_model.mean_ + np.dot(weights, self.pca_model.components_)
        path_final = save_array_as_image(final_vec.reshape(h, w), str(self.image_dir / "eq_final.png"))
        face_final = ImageMobject(path_final).scale(3).move_to(face_mob)
        
        self.play(Transform(face_mob, face_final), FadeOut(ff_text), face_label.animate.become(
            Text("Reconstructed Face", font_size=24, color=Colors.SUCCESS).next_to(face_mob, DOWN)
        ))
        self.wait(2)
        
        self.play(FadeOut(Group(title, face_mob, face_label)))
    
    # =========================================================================
    # OUTRO
    # =========================================================================
    
    def _outro_sequence(self):
        messages = VGroup(
            Text("Linear algebra gives us the language", font_size=36, color=Colors.PRIMARY),
            Text("to understand high-dimensional worlds.", font_size=36, color=Colors.PRIMARY),
        ).arrange(DOWN, buff=0.3)
        
        self.play(Write(messages), run_time=2)
        self.wait(3)
        self.play(FadeOut(messages, shift=UP * 0.5), run_time=1.5)


# =============================================================================
# RUN: manimgl linear_algebra_course.py LinearAlgebraCourse -o
# =============================================================================
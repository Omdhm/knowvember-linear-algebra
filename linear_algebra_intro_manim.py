from manimlib import *
import numpy as np
import os


# ---------- UTILS ----------
def integer_matrix_tex(A):
    """Make a nice IntegerMatrix if A is integral, else fallback to Matrix."""
    if np.allclose(A, np.round(A)):
        return IntegerMatrix(A.astype(int))
    return Matrix(A.round(2))

def vector_tex(v):
    return Tex(r"\begin{bmatrix}" + f"{v[0]:.1f}" + r"\\" + f"{v[1]:.1f}" + r"\end{bmatrix}")

def signed_area(a, b):
    # 2D signed area = det([a b])
    return a[0]*b[1] - a[1]*b[0]

def np2img_file(arr, path):
    """Save a grayscale numpy array to PNG and return path (uses PIL if available)."""
    try:
        from PIL import Image
        arr_clip = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr_clip).save(path)
        return path
    except Exception:
        # Fallback: write a simple PGM file Manim can still load via ImageMobject
        with open(path.replace(".png", ".pgm"), "wb") as f:
            h, w = arr.shape[:2]
            f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
            f.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
        return path.replace(".png", ".pgm")

# ---------- MAIN LESSON ----------
class LinearAlgebraLesson(InteractiveScene):
    def construct(self):
        self.section = 0
        self.next_section()

    def next_section(self):
        self.clear()
        sections = [
            self.section_vectors_and_spans,      # 0
            self.section_linear_transform,       # 1
            self.section_determinant,            # 2
            self.section_dot_duality_cross,      # 3
            self.section_change_basis_eigen,     # 4
            self.section_pca_lfw                 # 5
        ]
        if self.section < len(sections):
            sections[self.section]()
            self.section += 1
            self.next_section()
        else:
            end = Text("That's all. Thanks!").scale(0.9)
            if hasattr(end, "set_backstroke"):
                end.set_backstroke(width=5)
            self.play(Write(end))
            self.wait(1.5)

    # ---------- 1) Vectors & Spans ----------
    def section_vectors_and_spans(self):
        title = Text("1. Vectors and Spans").to_edge(UP)
        if hasattr(title, "set_backstroke"): title.set_backstroke(width=5)
        self.play(Write(title))
        plane = NumberPlane((-6, 6), (-4, 4))
        self.play(ShowCreation(plane))  # ManimGL example style

        # Two draggable vectors a, b
        a_tip = MotionMobject(Dot(2*RIGHT + 1*UP, color=YELLOW))
        b_tip = MotionMobject(Dot(1.5*RIGHT + 2*UP, color=YELLOW))
        a_arrow = Arrow(ORIGIN, a_tip.mobject.get_center(), buff=0, color=BLUE)
        b_arrow = Arrow(ORIGIN, b_tip.mobject.get_center(), buff=0, color=GREEN)

        a_arrow.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, a_tip.mobject.get_center()))
        b_arrow.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, b_tip.mobject.get_center()))

        # Parallelogram for span{a,b}
        pg = Polygon(ORIGIN, a_tip.mobject.get_center(),
                     a_tip.mobject.get_center() + (b_tip.mobject.get_center()),
                     b_tip.mobject.get_center())
        pg.set_fill(PURPLE, opacity=0.15).set_stroke(PURPLE, 2)

        def upd_pg(m):
            A = a_tip.mobject.get_center()
            B = b_tip.mobject.get_center()
            m.set_points_as_corners([ORIGIN, A, A+B, B, ORIGIN])
            m.make_smooth()
        pg.add_updater(upd_pg)

        # Live labels
        a_lab = Tex(r"\vec a=").next_to(3*LEFT + 3*UP, RIGHT)
        a_val = vector_tex(a_tip.mobject.get_center()[:2])
        b_lab = Tex(r"\vec b=").next_to(3*LEFT + 2*UP, RIGHT)
        b_val = vector_tex(b_tip.mobject.get_center()[:2])
        area_text = Tex(r"\text{Area}=\;|\det([a\;b])|").next_to(3*LEFT + 1*UP, RIGHT)

        def upd_val_A(m):
            x, y, _ = a_tip.mobject.get_center()
            m.become(vector_tex([x, y]).move_to(a_val.get_center()))
        def upd_val_B(m):
            x, y, _ = b_tip.mobject.get_center()
            m.become(vector_tex([x, y]).move_to(b_val.get_center()))
        a_val.add_updater(upd_val_A)
        b_val.add_updater(upd_val_B)

        self.add(a_arrow, b_arrow, pg, a_lab, a_val, b_lab, b_val, area_text, a_tip, b_tip)
        self.wait()

        # Clean up updaters before leaving the section
        a_arrow.clear_updaters(); b_arrow.clear_updaters(); pg.clear_updaters()
        a_val.clear_updaters(); b_val.clear_updaters()
        self.wait(0.2)

    # ---------- 2) Linear Transformation ----------
    def section_linear_transform(self):
        title = Text("2. Linear Transformation").to_edge(UP)
        if hasattr(title, "set_backstroke"): title.set_backstroke(width=5)
        self.play(Write(title))

        plane = NumberPlane((-6, 6), (-4, 4))
        self.play(ShowCreation(plane))
        A = np.array([[1.0, 1.0],
                      [0.0, 1.0]])  # shear

        e1 = Arrow(ORIGIN, RIGHT, buff=0)
        e2 = Arrow(ORIGIN, UP, buff=0)
        self.play(GrowArrow(e1), GrowArrow(e2))
        e1_label = Tex(r"\vec{e}_1").next_to(e1.get_end(), DOWN)
        e2_label = Tex(r"\vec{e}_2").next_to(e2.get_end(), LEFT)
        self.play(Write(VGroup(e1_label, e2_label)))
        mat = integer_matrix_tex(A).to_corner(UL)
        self.play(Write(mat))

        self.wait()
        # Either animate.apply_matrix or ApplyMatrix; both are used in ManimGL examples
        self.play(
            plane.animate.apply_matrix(A),
            VGroup(e1, e2, e1_label, e2_label).animate.apply_matrix(A),
            run_time=3
        )
        self.wait()

    # ---------- 3) Determinant ----------
    def section_determinant(self):
        title = Text("3. Determinant (Area scaling & orientation)").to_edge(UP)
        if hasattr(title, "set_backstroke"): title.set_backstroke(width=5)
        self.play(Write(title))

        plane = NumberPlane((-5, 5), (-3, 3))
        self.play(ShowCreation(plane))

        # Unit square -> parallelogram
        square = Polygon(ORIGIN, RIGHT, RIGHT + UP, UP, color=YELLOW).set_fill(YELLOW, 0.15)
        self.play(ShowCreation(square))

        # Two matrices: det>0 (rotation+scale) and det<0 (reflection)
        A = np.array([[1.2, 0.6],
                      [-0.2, 1.1]])
        B = np.array([[1.0, 0.0],
                      [0.0, -1.0]])  # reflection across x-axis: det(B) = -1

        detA = np.linalg.det(A)
        detB = np.linalg.det(B)
        tA = Tex(r"\det(A) \approx " + f"{detA:.2f}").to_corner(UL)
        tB = Tex(r"\det(B) = -1").next_to(tA, DOWN)

        self.play(Write(tA))
        self.wait()
        self.play(square.animate.apply_matrix(A), plane.animate.apply_matrix(A), run_time=3)

        self.play(Write(tB))
        self.wait()
        self.play(square.animate.apply_matrix(B), plane.animate.apply_matrix(B), run_time=2)
        self.wait()

    # ---------- 4) Dot, Duality, Cross ----------
    def section_dot_duality_cross(self):
        title = Text("4. Dot Product, Duality, Cross").to_edge(UP)
        if hasattr(title, "set_backstroke"): title.set_backstroke(width=5)
        self.play(Write(title))

        plane = NumberPlane((-6, 6), (-4, 4))
        self.play(ShowCreation(plane))

        a = Arrow(ORIGIN, 2*RIGHT, buff=0, color=BLUE)
        self.play(GrowArrow(a))
        a_lab = Tex(r"\vec a").next_to(a.get_end(), DOWN)

        # Draggable vector w
        w_tip = MotionMobject(Dot(2*RIGHT + 2*UP, color=YELLOW))
        w = Arrow(ORIGIN, w_tip.mobject.get_center(), buff=0, color=GREEN)
        w.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, w_tip.mobject.get_center()))

        proj = always_redraw(lambda: Line(
            a.get_start(),
            # projection scalar * unit(a)
            (np.dot(w_tip.mobject.get_center()[:2], np.array([2,0])) / np.linalg.norm([2,0])**2) * a.get_end(),
            color=WHITE
        ))
        dot_val = DecimalNumber(0, num_decimal_places=2).to_corner(UR)
        dot_text = Tex(r"\vec a \cdot \vec w =").to_corner(UR).shift(1.2*LEFT)

        def upd_dot(m):
            wv = w_tip.mobject.get_center()[:2]
            av = np.array([2.0, 0.0])
            m.set_value(float(np.dot(av, wv)))
        dot_val.add_updater(upd_dot)

        # Duality: covector α defines lines α·x = c (parallel level sets)
        alpha = np.array([1.0, -0.5])
        c_tracker = ValueTracker(0.0)

        def level_line(c):
            # α·x = c  => y = (c - αx)/αy  (if αy != 0)
            if abs(alpha[1]) < 1e-6:
                x = c/alpha[0]
                return Line(np.array([x, -10, 0]), np.array([x, 10, 0]), color=RED)
            else:
                def y(x): return (c - alpha[0]*x)/alpha[1]
                P = np.array([-10, y(-10), 0]); Q = np.array([10, y(10), 0])
                return Line(P, Q, color=RED)
        level = always_redraw(lambda: level_line(c_tracker.get_value()))

        # Cross product magnitude in 2D == area (|det([a w])|)
        area_label = always_redraw(lambda:
            Tex(r"|a \times w| = \text{area} = | \det([a\;w]) |").next_to(plane, DOWN)
        )

        self.add(a, a_lab, w, w_tip, proj, dot_text, dot_val, level, area_label)
        self.wait()

        # Slide c to show parallel lines (duality view)
        self.play(c_tracker.animate.set_value(2.0), run_time=1.5)
        self.play(c_tracker.animate.set_value(-2.0), run_time=1.5)
        self.wait()

        w.clear_updaters()

    # ---------- 5) Change of Basis & Eigen ----------
    def section_change_basis_eigen(self):
        title = Text("5. Change of Basis & Eigenvectors").to_edge(UP)
        if hasattr(title, "set_backstroke"): title.set_backstroke(width=5)
        self.play(Write(title))

        plane = NumberPlane((-6, 6), (-4, 4))
        self.play(ShowCreation(plane))

        # New basis B columns
        b1 = np.array([2.0, 1.0])
        b2 = np.array([1.0, 2.0])
        B = np.column_stack([b1, b2])  # 2x2
        b1_arr = Arrow(ORIGIN, b1[0]*RIGHT + b1[1]*UP, buff=0, color=BLUE)
        b2_arr = Arrow(ORIGIN, b2[0]*RIGHT + b2[1]*UP, buff=0, color=GREEN)
        self.play(GrowArrow(b1_arr), GrowArrow(b2_arr))
        B_tex = integer_matrix_tex(B).to_corner(UL)
        self.play(Write(B_tex))

        # A sample vector x and its coords in B (x = B c)
        c = np.array([1.5, -0.5])
        x = B @ c
        x_arr = Arrow(ORIGIN, x[0]*RIGHT + x[1]*UP, buff=0, color=YELLOW)
        self.play(GrowArrow(x_arr))
        x_std = Tex(r"\vec x_{\text{std}}=").next_to(3*LEFT + 2.8*UP, RIGHT)
        x_std_val = vector_tex(x)
        x_B = Tex(r"[\vec x]_{B}=").next_to(3*LEFT + 1.9*UP, RIGHT)
        x_B_val = vector_tex(c)
        self.add(x_std, x_std_val, x_B, x_B_val)

        self.wait()

        # A with two real eigenvectors
        A = np.array([[2.0, 1.0],
                      [1.0, 2.0]])
        evals, evecs = np.linalg.eig(A)
        evecs = evecs / np.linalg.norm(evecs, axis=0, keepdims=True)
        e1 = evecs[:, 0]; e2 = evecs[:, 1]
        ev1 = Arrow(ORIGIN, 3*(e1[0]*RIGHT + e1[1]*UP), color=RED)
        ev2 = Arrow(ORIGIN, 3*(e2[0]*RIGHT + e2[1]*UP), color=RED)
        e_lab = Tex(r"\text{eigenvectors}").next_to(ev2, RIGHT)
        A_tex = integer_matrix_tex(A).to_corner(UR)

        self.play(Write(A_tex), GrowArrow(ev1), GrowArrow(ev2), Write(e_lab))
        self.wait()
        self.play(
            plane.animate.apply_matrix(A),
            VGroup(b1_arr, b2_arr, x_arr, ev1, ev2).animate.apply_matrix(A),
            run_time=3
        )
        self.wait()

    # ---------- 6) PCA with LFW ----------
    def section_pca_lfw(self):
        title = Text("6. PCA (Eigenfaces) – LFW").to_edge(UP)
        if hasattr(title, "set_backstroke"): title.set_backstroke(width=5)
        self.play(Write(title))

        note = Text("Loading LFW (scikit-learn) ...", font="Consolas").scale(0.5).to_corner(UL)
        self.add(note)

        # Try to fetch an LFW subset and compute PCA; provide fallbacks.
        face_path = None
        mean_path = None
        eig_paths = []
        try:
            from sklearn.datasets import fetch_lfw_people
            from sklearn.decomposition import PCA
            # Cache under project folder to avoid restricted home dirs
            data_home = os.path.join(os.getcwd(), "sklearn_data")
            faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                                     color=False, data_home=data_home, download_if_missing=True)
            # X: (n_samples, n_features), images: (n_samples, h, w)
            X = faces.data
            h, w = faces.images.shape[1], faces.images.shape[2]
            # Save one face
            x0 = faces.images[0] * 255.0  # images are floats in [0,1]; write as [0,255]
            face_path = np2img_file(x0, "lfw_face.png")

            # PCA (Eigenfaces) following sklearn example (randomized + whiten)
            n_components = 16
            pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X)
            mean_img = pca.mean_.reshape(h, w) * 255.0
            mean_path = np2img_file(mean_img, "lfw_mean.png")
            comps = pca.components_.reshape((n_components, h, w))
            for i in range(8):  # first 8 eigenfaces
                eig_paths.append(np2img_file(255.0 * (comps[i] - comps[i].min()) /
                                             (comps[i].ptp() + 1e-8), f"lfw_eig_{i}.png"))

            # Reconstructions with k=10 and k=50 (if feasible)
            k1, k2 = 10, min(50, n_components)
            # Fit another PCA with more components if needed
            if k2 > n_components:
                pca2 = PCA(n_components=k2, svd_solver="randomized", whiten=True).fit(X)
                comps2 = pca2.components_
                x0_flat = X[0].reshape(1, -1)
                rec1 = (pca2.inverse_transform(pca2.transform(x0_flat)[:,:k1])
                        .reshape(h, w))
                rec2 = (pca2.inverse_transform(pca2.transform(x0_flat)[:,:k2])
                        .reshape(h, w))
            else:
                x0_flat = X[0].reshape(1, -1)
                rec1 = (pca.inverse_transform(pca.transform(x0_flat)[:,:k1]).reshape(h, w))
                rec2 = (pca.inverse_transform(pca.transform(x0_flat)[:,:k2]).reshape(h, w))
            rec1_path = np2img_file(255.0 * rec1, "lfw_rec_10.png")
            rec2_path = np2img_file(255.0 * rec2, "lfw_rec_50.png")
        except Exception as e:
            # Fallback: show a synthetic grayscale gradient if sklearn or download not available
            grad = np.tile(np.linspace(0, 255, 200), (200,1))
            face_path = np2img_file(grad, "lfw_face_fallback.png")
            mean_path = np2img_file(grad, "lfw_mean_fallback.png")
            eig_paths = [np2img_file(grad.T, "lfw_eig_fallback.png")]

        self.remove(note)

        # Lay out the visuals
        plane = NumberPlane((-7, 7), (-4, 4))
        self.play(ShowCreation(plane))

        face_img = ImageMobject(face_path).scale(1.5).to_corner(UL)
        self.play(FadeIn(face_img))
        self.wait()

        mean_imgm = ImageMobject(mean_path).scale(1.5).next_to(face_img, RIGHT, buff=1)
        self.play(FadeIn(mean_imgm))
        self.wait()

        # Show first 4 eigenfaces in a row
        row = VGroup(*[ImageMobject(p).scale(1.2) for p in eig_paths[:4]])
        row.arrange(RIGHT, buff=0.6).next_to(mean_imgm, DOWN, buff=0.8).align_to(face_img, LEFT)
        label = Text("Top eigenfaces").scale(0.5).next_to(row, UP)
        self.play(Write(label), FadeIn(row))
        self.wait()

        # Reconstructions if we produced them
        recs = []
        if os.path.exists("lfw_rec_10.png") and os.path.exists("lfw_rec_50.png"):
            r10 = ImageMobject("lfw_rec_10.png").scale(1.5).to_corner(DL).shift(1.2*UP)
            r50 = ImageMobject("lfw_rec_50.png").scale(1.5).next_to(r10, RIGHT, buff=1.0)
            t10 = Text("k=10").scale(0.5).next_to(r10, DOWN, buff=0.2)
            t50 = Text("k=50").scale(0.5).next_to(r50, DOWN, buff=0.2)
            recs = [r10, r50, t10, t50]
            self.play(FadeIn(VGroup(*recs)))
            self.wait()
        else:
            self.wait()
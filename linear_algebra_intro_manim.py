from manimlib import *
import numpy as np

# Scene 1: vectors on the plane, using ManimGL patterns
class VectorIntro(InteractiveScene):
    def construct(self):
        title = Text("Vectors as arrows in the plane")
        title.to_edge(UP)
        self.play(Write(title))  # ManimGL: Write / ShowCreation are standard [6](https://stackoverflow.com/questions/72121902/numberplane-grid-not-showing-manimgl-scene)

        plane = NumberPlane((-6, 6), (-4, 4))
        self.play(ShowCreation(plane))  # ShowCreation is used in ManimGL examples [6](https://stackoverflow.com/questions/72121902/numberplane-grid-not-showing-manimgl-scene)

        v = np.array([2, 1])
        vec = Arrow(ORIGIN, v[0] * RIGHT + v[1] * UP, buff=0)
        self.play(GrowArrow(vec))  # GrowArrow exists in ManimGL [5](https://github.com/3b1b/manim/blob/master/manimlib/animation/growing.py)

        label = Tex(r"\vec{v} = \begin{bmatrix}2\\1\end{bmatrix}")
        # In ManimGL, position labels relative to mobjects/points explicitly
        label.move_to(vec.get_end() + 0.3 * UP + 0.15 * RIGHT)
        self.play(Write(label))
        self.wait(1.5)

        # Slide the vector around to show "same vector" idea
        self.play(Transform(vec, vec.copy().shift(LEFT + DOWN)))
        self.wait(1)
        self.play(Transform(vec, vec.copy().shift(RIGHT + UP)))
        self.wait(1.5)


# Scene 2: a 2D linear transformation on plane and basis
class LinearTransform2D(Scene):
    def construct(self):
        title = Text("A matrix as a transformation")
        title.to_edge(UP)
        self.play(Write(title))

        plane = NumberPlane((-6, 6), (-4, 4))
        self.play(ShowCreation(plane))  # Pattern used in OpeningManimExample [6](https://stackoverflow.com/questions/72121902/numberplane-grid-not-showing-manimgl-scene)

        # Simple shear
        A = np.array([[1.0, 1.0],
                      [0.0, 1.0]])

        # Basis vectors
        e1 = Arrow(ORIGIN, RIGHT, buff=0)
        e2 = Arrow(ORIGIN, UP, buff=0)
        basis_vecs = VGroup(e1, e2)
        self.play(*[GrowArrow(v) for v in basis_vecs])  # GrowArrow in GL [5](https://github.com/3b1b/manim/blob/master/manimlib/animation/growing.py)

        # Place basis labels near arrow tips (relative to mobjects, not constants)
        e1_label = Tex(r"\vec{e}_1").next_to(e1.get_end(), DOWN)
        e2_label = Tex(r"\vec{e}_2").next_to(e2.get_end(), LEFT)
        self.play(Write(VGroup(e1_label, e2_label)))

        # Show the matrix on screen, prefer IntegerMatrix (matches 3b1b examples)
        matrix_tex = IntegerMatrix(A.astype(int))
        matrix_tex.to_corner(UL)
        self.play(Write(matrix_tex))  # ManimGL uses Write / FadeTransform frequently [6](https://stackoverflow.com/questions/72121902/numberplane-grid-not-showing-manimgl-scene)

        # Animate the plane and basis under A
        # Either ApplyMatrix(...) or the newer .animate.apply_matrix(...) both exist in ManimGL.
        # Grant’s example uses .animate.apply_matrix on NumberPlane. [6](https://stackoverflow.com/questions/72121902/numberplane-grid-not-showing-manimgl-scene)
        self.play(
            plane.animate.apply_matrix(A),
            basis_vecs.animate.apply_matrix(A),
            run_time=3
        )
        self.wait(2)


# Scene 3: PCA-style projection of a correlated point cloud
class PCAProjection(Scene):
    def construct(self):
        title = Text("Projecting data onto a direction")
        title.to_edge(UP)
        self.play(Write(title))

        plane = NumberPlane((-6, 6), (-4, 4))
        self.play(ShowCreation(plane))  # Consistent with GL examples [6](https://stackoverflow.com/questions/72121902/numberplane-grid-not-showing-manimgl-scene)

        # Fake correlated 2D dataset
        np.random.seed(1)
        points = []
        for _ in range(40):
            x = np.random.normal(0, 2)
            y = 0.8 * x + np.random.normal(0, 0.5)
            points.append(np.array([x, y, 0]))

        dots = VGroup(*[Dot(pt) for pt in points])  # Dot(point) signature in GL examples
        self.play(FadeIn(dots))
        self.wait(1)

        # Principal direction (unit)
        u = np.array([1.0, 0.8])
        u = u / np.linalg.norm(u)

        direction_line = Line(10 * np.array([-u[0], -u[1], 0]),
                              10 * np.array([u[0], u[1], 0]),
                              color=YELLOW)
        self.play(ShowCreation(direction_line))
        self.wait(1)

        # Orthogonal projections onto direction u
        projection_dots = VGroup()
        projection_segments = VGroup()
        for pt in points:
            p2d = np.array([pt[0], pt[1]])
            scalar = np.dot(p2d, u)
            proj_2d = scalar * u
            proj = np.array([proj_2d[0], proj_2d[1], 0])
            projection_dots.add(Dot(proj, color=YELLOW))
            projection_segments.add(Line(pt, proj, stroke_opacity=0.5))

        self.play(
            ShowCreation(projection_segments),
            FadeIn(projection_dots),
            run_time=3
        )
        self.wait(2)
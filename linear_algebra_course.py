from manimlib import *
import numpy as np

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
# MAIN CLASS
# =============================================================================

class LinearAlgebraCourse(ThreeDScene):
    
    def construct(self):
        self._setup_resources()
        
    #    self._intro_sequence()
     #   self._chapter_vectors()
       # self._chapter_linear_combinations()
       # self._chapter_transformations()
       # self._chapter_matrix_multiplication()
       # self._chapter_determinants()
        self._chapter_dot_product()
        self._chapter_cross_product()
        self._chapter_eigenvectors()
        self._outro_sequence()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    def _setup_resources(self):
        """Initialize any resources needed for the animations."""
        pass
    
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

        
        title_mob = Text(title, font_size=Config.TITLE_FONT_SIZE - 12, color=Colors.TEXT_PRIMARY)
        group = VGroup( title_mob)
        
        self.play( FadeIn(title_mob, shift=UP * 0.2), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(group), run_time=0.5)
    
    def _create_grid(self, opacity: float = 0.4, color=None):
        color = color or Colors.GRID_MAIN
        plane = NumberPlane(
            x_range=(-7, 7, 1), y_range=(-5, 5, 1),
            background_line_style={"stroke_color": color, "stroke_width": 1, "stroke_opacity": opacity * 0.5},
            axis_config={"stroke_color": color, "stroke_width": 2, "stroke_opacity": opacity}
        )
        plane.set_z_index(-10)  # Put grid behind other elements
        return plane
    
    # =========================================================================
    # INTRO
    # =========================================================================
    
    def _intro_sequence(self):
        # 3D Intro
        title = Text("Linear Algebra", font_size=Config.TITLE_FONT_SIZE, color=Colors.PRIMARY, weight=BOLD)
        subtitle = Text("An Intuitive understanding", font_size=Config.SUBTITLE_FONT_SIZE, color=Colors.TEXT_SECONDARY)
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
        
    

    
    # =========================================================================
    # CHAPTER 1: VECTORS - "What even are they?"
    # Based on 3Blue1Brown's Essence of Linear Algebra
    # =========================================================================
    
    def _chapter_vectors(self):
        self._create_chapter_header(1, "Vectors, what even are they?")
        
        # --- PART 1: Thinking About Coordinate Systems ---
        self._vectors_coordinate_system()
        
        # --- PART 2: Vectors as Arrows with Coordinates ---
        self._vectors_as_arrows()
        
        # --- PART 3: Vector Addition ---
        self._vectors_addition()
        
        # --- PART 4: Scalar Multiplication ---
        self._vectors_scaling()
    
    def _vectors_coordinate_system(self):
        """Introduce the coordinate system - the stage for vectors."""
        
        # Start with just the axes
        intro_text = Text("Let's start by thinking about coordinate systems", 
                         font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        intro_text.to_edge(UP, buff=0.5)
        self.play(Write(intro_text))
        self.wait(1)
        
        # Draw x-axis
        x_axis = Line(LEFT * 6, RIGHT * 6, color=Colors.GRID_MAIN, stroke_width=3)
        x_label = Text("x-axis", font_size=24, color=Colors.TEXT_SECONDARY)
        x_label.next_to(x_axis, RIGHT, buff=0.2)
        
        self.play(ShowCreation(x_axis), Write(x_label))
        
        # Draw y-axis  
        y_axis = Line(DOWN * 4, UP * 4, color=Colors.GRID_MAIN, stroke_width=3)
        y_label = Text("y-axis", font_size=24, color=Colors.TEXT_SECONDARY)
        y_label.next_to(y_axis, UP, buff=0.2)
        
        self.play(ShowCreation(y_axis), Write(y_label))
        
        # Highlight the origin
        origin_dot = Dot(ORIGIN, color=Colors.ACCENT, radius=0.12)
        origin_label = Text("Origin", font_size=24, color=Colors.ACCENT)
        origin_label.next_to(origin_dot, DL, buff=0.2)
        
        self.play(FadeOut(intro_text))
        origin_text = Text("The origin is the center of space, the root of all vectors",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        origin_text.to_edge(UP, buff=0.5)
        self.play(Write(origin_text))
        self.play(GrowFromCenter(origin_dot), Write(origin_label))
        self.wait(1.5)
        
        # Add tick marks representing unit length
        self.play(FadeOut(origin_text))
        unit_text = Text("Choose a distance to represent 1 unit",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        unit_text.to_edge(UP, buff=0.5)
        self.play(Write(unit_text))
        
        # Show unit tick marks on axes
        tick_marks = VGroup()
        for i in range(-5, 6):
            if i != 0:
                x_tick = Line(UP * 0.1, DOWN * 0.1, color=Colors.GRID_MAIN, stroke_width=2)
                x_tick.move_to([i, 0, 0])
                y_tick = Line(LEFT * 0.1, RIGHT * 0.1, color=Colors.GRID_MAIN, stroke_width=2)
                y_tick.move_to([0, i, 0])
                tick_marks.add(x_tick, y_tick)
        
        self.play(ShowCreation(tick_marks, lag_ratio=0.05))
        self.wait(1)
        
        # Extend to full grid
        self.play(FadeOut(unit_text))
        grid_text = Text("Extend these to create a grid",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        grid_text.to_edge(UP, buff=0.5)
        self.play(Write(grid_text))
        
        plane = self._create_grid()
        self.play(
            FadeOut(x_axis), FadeOut(y_axis), FadeOut(tick_marks),
            FadeOut(x_label), FadeOut(y_label),
            ShowCreation(plane, lag_ratio=0.02),
            run_time=2
        )
        self.wait(1)
        
        # Keep origin dot and plane for next section
        self.play(FadeOut(grid_text), FadeOut(origin_label))
        
        # Store for later use
        self.current_plane = plane
        self.origin_dot = origin_dot
    
    def _vectors_as_arrows(self):
        """Vectors as arrows from origin with coordinates as instructions."""
        
        # Explain vectors as arrows
        arrow_text = Text("A vector is an arrow inside this coordinate system",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        arrow_text.to_edge(UP, buff=0.5)
        self.play(Write(arrow_text))
        
        # Draw a vector
        v = Vector([3, 2], color=Colors.RESULT, stroke_width=5)
        self.play(GrowArrow(v))
        self.wait(1)
        
        # Explain tail at origin
        self.play(FadeOut(arrow_text))
        tail_text = Text("Its tail sits at the origin",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        tail_text.to_edge(UP, buff=0.5)
        self.play(Write(tail_text))
        self.play(Indicate(self.origin_dot, color=Colors.ACCENT, scale_factor=1.5))
        self.wait(1)
        
        # Coordinates as instructions
        self.play(FadeOut(tail_text))
        coord_text = Text("Coordinates are instructions to reach the tip",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        coord_text.to_edge(UP, buff=0.5)
        self.play(Write(coord_text))
        self.wait(1)
        
        # Show the walk: first along x, then along y
        # Horizontal component
        h_line = DashedLine(ORIGIN, [3, 0, 0], color=Colors.PRIMARY, stroke_width=3)
        h_label = Text("3 right", font_size=24, color=Colors.PRIMARY)
        h_label.next_to(h_line, DOWN, buff=0.2)
        
        self.play(ShowCreation(h_line), Write(h_label))
        self.wait(0.5)
        
        # Vertical component
        v_line = DashedLine([3, 0, 0], [3, 2, 0], color=Colors.SECONDARY, stroke_width=3)
        v_label = Text("2 up", font_size=24, color=Colors.SECONDARY)
        v_label.next_to(v_line, RIGHT, buff=0.2)
        
        self.play(ShowCreation(v_line), Write(v_label))
        self.wait(1)
        
        # Show coordinate notation
        self.play(FadeOut(coord_text))
        notation_text = Text("We write this as a column of numbers",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        notation_text.to_edge(UP, buff=0.5)
        self.play(Write(notation_text))
        
        coords = Tex(r"\begin{bmatrix} 3 \\ 2 \end{bmatrix}",
                    font_size=48, color=Colors.RESULT)
        coords.next_to(v.get_end(), UR, buff=0.3)
        self.play(Write(coords))
        self.wait(1.5)
        
        # Explain the mapping
        self.play(FadeOut(notation_text))
        mapping_text = Text("Every pair of numbers ↔ one unique vector",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        mapping_text.to_edge(UP, buff=0.5)
        self.play(Write(mapping_text))
        self.wait(2)
        
        # Clean up for next section
        self.play(
            FadeOut(v), FadeOut(coords), FadeOut(h_line), FadeOut(v_line),
            FadeOut(h_label), FadeOut(v_label), FadeOut(mapping_text)
        )
    
    def _vectors_addition(self):
        """Vector addition using tip-to-tail method."""
        
        # Title
        add_title = Text("Vector Addition",
                        font_size=Config.SUBTITLE_FONT_SIZE, color=Colors.PRIMARY)
        add_title.to_edge(UP, buff=0.5)
        self.play(Write(add_title))
        self.wait(0.5)
        
        # Show two vectors - use larger vectors to avoid crowding
        v1 = Vector([2, 3], color=Colors.PRIMARY, stroke_width=4)
        v2_at_origin = Vector([3, -1], color=Colors.SECONDARY, stroke_width=4)
        
        # Place labels away from vectors to avoid overlap
        label_v1 = Tex(r"\vec{a} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}", color=Colors.PRIMARY, font_size=32)
        label_v1.to_corner(UL, buff=1).shift(DOWN * 1.5)
        
        label_v2 = Tex(r"\vec{b} = \begin{bmatrix} 3 \\ -1 \end{bmatrix}", color=Colors.SECONDARY, font_size=32)
        label_v2.next_to(label_v1, DOWN, buff=0.4)
        
        self.play(GrowArrow(v1), Write(label_v1))
        self.play(GrowArrow(v2_at_origin), Write(label_v2))
        self.wait(1)
        
        # Explain tip-to-tail
        self.play(FadeOut(add_title))
        tip_tail_text = Text("Move the second vector's tail to the first's tip",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        tip_tail_text.to_edge(UP, buff=0.5)
        self.play(Write(tip_tail_text))
        
        # Animate moving v2 to tip of v1
        v2_moved = Vector([3, -1], color=Colors.SECONDARY, stroke_width=4)
        v2_moved.shift(v1.get_end())
        
        self.play(
            Transform(v2_at_origin, v2_moved),
            run_time=1.5
        )
        self.wait(1)
        
        # Draw the sum vector
        self.play(FadeOut(tip_tail_text))
        sum_text = Text("The sum is a new vector from origin to the final tip",
                       font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        sum_text.to_edge(UP, buff=0.5)
        self.play(Write(sum_text))
        
        v_sum = Vector([5, 2], color=Colors.SUCCESS, stroke_width=5)
        self.play(GrowArrow(v_sum))
        
        # Show the sum coordinates - place in corner to avoid overlap
        sum_label = Tex(r"\vec{a} + \vec{b} = \begin{bmatrix} 2+3 \\ 3+(-1) \end{bmatrix} = \begin{bmatrix} 5 \\ 2 \end{bmatrix}",
                       color=Colors.SUCCESS, font_size=32)
        sum_label.next_to(label_v2, DOWN, buff=0.4)
        self.play(Write(sum_label))
        self.wait(1)
        
        # Show the formula
        self.play(FadeOut(sum_text))
        formula_text = Tex(r"\begin{bmatrix} x_1 \\ y_1 \end{bmatrix} + \begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} x_1 + x_2 \\ y_1 + y_2 \end{bmatrix}",
                          font_size=36, color=Colors.TEXT_PRIMARY)
        formula_text.to_edge(UP, buff=0.5)
        self.play(Write(formula_text))
        self.wait(2)
        
        # Why does this make sense? Movement analogy
        self.play(FadeOut(formula_text))
        movement_text = Text("Think of each vector as a movement or step",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        movement_text.to_edge(UP, buff=0.5)
        self.play(Write(movement_text))
        self.wait(1.5)
        
        self.play(FadeOut(movement_text))
        analogy_text = Text("Walking along both gives the same result as the sum",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        analogy_text.to_edge(UP, buff=0.5)
        self.play(Write(analogy_text))
        self.play(Indicate(v_sum, color=Colors.SUCCESS, scale_factor=1.1))
        self.wait(2)
        
        # Clean up
        self.play(
            FadeOut(v1), FadeOut(v2_at_origin), FadeOut(v_sum),
            FadeOut(label_v1), FadeOut(label_v2), FadeOut(sum_label),
            FadeOut(analogy_text)
        )
    
    def _vectors_scaling(self):
        """Scalar multiplication - stretching, squishing, flipping vectors."""
        
        # Title
        scale_title = Text("Scalar Multiplication",
                          font_size=Config.SUBTITLE_FONT_SIZE, color=Colors.PRIMARY)
        scale_title.to_edge(UP, buff=0.5)
        self.play(Write(scale_title))
        
        # Original vector
        v_original = Vector([2, 1], color=Colors.RESULT, stroke_width=4)
        v_label = Tex(r"\vec{v} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}",
                     font_size=32, color=Colors.RESULT)
        v_label.next_to(v_original.get_end(), UR, buff=0.2)
        
        self.play(GrowArrow(v_original), Write(v_label))
        self.wait(1)
        
        # Scaling by 2 - stretch
        self.play(FadeOut(scale_title))
        stretch_text = Text("Multiply by 2: stretch to twice the length",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        stretch_text.to_edge(UP, buff=0.5)
        self.play(Write(stretch_text))
        
        v_scaled_2 = Vector([4, 2], color=Colors.ACCENT, stroke_width=4)
        scaled_label = Tex(r"2\vec{v} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}",
                          font_size=32, color=Colors.ACCENT)
        scaled_label.next_to(v_scaled_2.get_end(), UR, buff=0.2)
        
        self.play(
            Transform(v_original, v_scaled_2),
            Transform(v_label, scaled_label),
            run_time=1.5
        )
        self.wait(1.5)
        
        # Reset
        v_original_reset = Vector([2, 1], color=Colors.RESULT, stroke_width=4)
        v_label_reset = Tex(r"\vec{v} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}",
                           font_size=32, color=Colors.RESULT)
        v_label_reset.next_to(v_original_reset.get_end(), UR, buff=0.2)
        
        self.play(
            Transform(v_original, v_original_reset),
            Transform(v_label, v_label_reset),
        )
        
        # Scaling by 1/3 - squish
        self.play(FadeOut(stretch_text))
        squish_text = Text("Multiply by 1/3: squish to one-third",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        squish_text.to_edge(UP, buff=0.5)
        self.play(Write(squish_text))
        
        v_scaled_third = Vector([2/3, 1/3], color=Colors.SECONDARY, stroke_width=4)
        third_label = Tex(r"\frac{1}{3}\vec{v} = \begin{bmatrix} 2/3 \\ 1/3 \end{bmatrix}",
                         font_size=32, color=Colors.SECONDARY)
        third_label.next_to(v_scaled_third.get_end(), UR, buff=0.2)
        
        self.play(
            Transform(v_original, v_scaled_third),
            Transform(v_label, third_label),
            run_time=1.5
        )
        self.wait(1.5)
        
        # Reset again
        self.play(
            Transform(v_original, v_original_reset),
            Transform(v_label, v_label_reset),
        )
        
        # Scaling by -1.5 - flip and stretch
        self.play(FadeOut(squish_text))
        flip_text = Text("Multiply by -1.5: flip direction and stretch",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        flip_text.to_edge(UP, buff=0.5)
        self.play(Write(flip_text))
        
        v_scaled_neg = Vector([-3, -1.5], color=Colors.ERROR, stroke_width=4)
        neg_label = Tex(r"-1.5\vec{v} = \begin{bmatrix} -3 \\ -1.5 \end{bmatrix}",
                       font_size=32, color=Colors.ERROR)
        neg_label.next_to(v_scaled_neg.get_end(), DL, buff=0.2)
        
        self.play(
            Transform(v_original, v_scaled_neg),
            Transform(v_label, neg_label),
            run_time=1.5
        )
        self.wait(1.5)
        
        # Explain "scalar"
        self.play(FadeOut(flip_text))
        scalar_text = Text("This process is called 'scaling' - hence 'scalar'",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        scalar_text.to_edge(UP, buff=0.5)
        self.play(Write(scalar_text))
        self.wait(1.5)
        
        # Show the formula
        self.play(FadeOut(scalar_text))
        formula_text = Tex(r"c \cdot \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} cx \\ cy \end{bmatrix}",
                          font_size=40, color=Colors.TEXT_PRIMARY)
        formula_text.to_edge(UP, buff=0.5)
        self.play(Write(formula_text))
        self.wait(2)
        
        # Conclusion
        self.play(FadeOut(formula_text))
        conclusion = Text("Vector addition and scalar multiplication are fundamental",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        conclusion.to_edge(UP, buff=0.5)
        self.play(Write(conclusion))
        self.wait(1)
        
        conclusion2 = Text("Every topic in linear algebra revolves around them",
                          font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        conclusion2.next_to(conclusion, DOWN, buff=0.3)
        self.play(Write(conclusion2))
        self.wait(2)
        
        # Final cleanup
        self.play(
            FadeOut(self.current_plane), FadeOut(self.origin_dot),
            FadeOut(v_original), FadeOut(v_label),
            FadeOut(conclusion), FadeOut(conclusion2)
        )

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
    # CHAPTER 2: LINEAR COMBINATIONS, SPAN, AND BASIS VECTORS
    # Based on 3Blue1Brown's Essence of Linear Algebra
    # =========================================================================
    
    def _chapter_linear_combinations(self):
        self._create_chapter_header(2, "Linear Combinations, Span & Basis")
        
        # --- PART 1: Coordinates as Scalars ---
        self._lincomb_coords_as_scalars()
        
        # --- PART 2: Basis Vectors ---
        self._lincomb_basis_vectors()
        
        # --- PART 3: Linear Combinations ---
        self._lincomb_combinations()
        
        # --- PART 4: Span (2D) ---
        self._lincomb_span_2d()
        
        # --- PART 5: Linear Dependence ---
        self._lincomb_linear_dependence()
    
    def _lincomb_coords_as_scalars(self):
        """Think of coordinates as scalars that scale basis vectors."""
        
        plane = self._create_grid()
        self.play(FadeIn(plane), run_time=0.8)
        
        # Introduction
        intro_text = Text("A new way to think about coordinates...",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        intro_text.to_edge(UP, buff=0.5)
        self.play(Write(intro_text))
        self.wait(1)
        
        # Show coordinate pair
        self.play(FadeOut(intro_text))
        coord_text = Text("Consider the vector (3, -2)",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        coord_text.to_edge(UP, buff=0.5)
        self.play(Write(coord_text))
        
        # Show the vector
        v = Vector([3, -2], color=Colors.RESULT, stroke_width=5)
        v_label = Tex(r"\begin{bmatrix} 3 \\ -2 \end{bmatrix}", color=Colors.RESULT, font_size=36)
        v_label.next_to(v.get_end(), DR, buff=0.2)
        self.play(GrowArrow(v), Write(v_label))
        self.wait(1)
        
        # Think of each coordinate as a scalar
        self.play(FadeOut(coord_text))
        scalar_text = Text("Think of each coordinate as a scalar",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        scalar_text.to_edge(UP, buff=0.5)
        self.play(Write(scalar_text))
        
        scalar_text2 = Text("...that stretches or squishes a vector",
                           font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        scalar_text2.next_to(scalar_text, DOWN, buff=0.2)
        self.play(Write(scalar_text2))
        self.wait(2)
        
        # Store plane for later, cleanup the rest
        self.lc_plane = plane
        self.play(FadeOut(v), FadeOut(v_label), FadeOut(scalar_text), FadeOut(scalar_text2))
    
    def _lincomb_basis_vectors(self):
        """Introduce i-hat and j-hat as basis vectors."""
        
        # Two special vectors
        special_text = Text("Two special vectors in the xy-coordinate system",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        special_text.to_edge(UP, buff=0.5)
        self.play(Write(special_text))
        
        # i-hat: unit vector along x
        i_hat = Vector([1, 0], color=Colors.I_HAT, stroke_width=6)
        i_label = Tex(r"\hat{\imath}", color=Colors.I_HAT, font_size=44)
        i_label.next_to(i_hat.get_end(), DOWN, buff=0.15)
        
        self.play(GrowArrow(i_hat), Write(i_label))
        
        i_desc = Tex(r"\hat{\imath} = \textrm{``i-hat''}", font_size=28, color=Colors.I_HAT)
        i_desc2 = Text("unit vector in x-direction", font_size=22, color=Colors.TEXT_SECONDARY)
        i_group = VGroup(i_desc, i_desc2).arrange(DOWN, buff=0.1)
        i_group.to_corner(UL, buff=1).shift(DOWN * 1.5)
        self.play(Write(i_group))
        self.wait(0.5)
        
        # j-hat: unit vector along y
        j_hat = Vector([0, 1], color=Colors.J_HAT, stroke_width=6)
        j_label = Tex(r"\hat{\jmath}", color=Colors.J_HAT, font_size=44)
        j_label.next_to(j_hat.get_end(), LEFT, buff=0.15)
        
        self.play(GrowArrow(j_hat), Write(j_label))
        
        j_desc = Tex(r"\hat{\jmath} = \textrm{``j-hat''}", font_size=28, color=Colors.J_HAT)
        j_desc2 = Text("unit vector in y-direction", font_size=22, color=Colors.TEXT_SECONDARY)
        j_group = VGroup(j_desc, j_desc2).arrange(DOWN, buff=0.1)
        j_group.next_to(i_group, DOWN, buff=0.4)
        self.play(Write(j_group))
        self.wait(1)
        
        # These together are called the "basis"
        self.play(FadeOut(special_text))
        basis_text = Text("Together, these are called the 'basis'",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        basis_text.to_edge(UP, buff=0.5)
        self.play(Write(basis_text))
        
        basis_text2 = Text("of the coordinate system",
                          font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        basis_text2.next_to(basis_text, DOWN, buff=0.2)
        self.play(Write(basis_text2))
        self.wait(2)
        
        # Store for next section
        self.lc_i_hat = i_hat
        self.lc_j_hat = j_hat
        self.lc_i_label = i_label
        self.lc_j_label = j_label
        
        # Fade out text descriptions but keep basis vectors visible
        self.play(FadeOut(basis_text), FadeOut(basis_text2), FadeOut(i_group), FadeOut(j_group))
        
        # Fade the basis vectors to make room for the demo
        self.play(
            self.lc_i_hat.animate.set_opacity(0.3),
            self.lc_j_hat.animate.set_opacity(0.3),
            self.lc_i_label.animate.set_opacity(0.3),
            self.lc_j_label.animate.set_opacity(0.3),
        )
    
    def _lincomb_combinations(self):
        """Show how any vector is a linear combination of basis vectors."""
        
        # Building a vector from basis vectors
        build_text = Text("Building (3, -2) from basis vectors",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        build_text.to_edge(UP, buff=0.5)
        self.play(Write(build_text))
        self.wait(0.5)
        
        # Step 1: Scale i-hat by 3
        self.play(FadeOut(build_text))
        step1_text = Text("x-coordinate (3) scales i-hat by 3",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.I_HAT)
        step1_text.to_edge(UP, buff=0.5)
        self.play(Write(step1_text))
        
        # Animate i-hat scaling - use TransformFromCopy properly
        i_scaled = Vector([3, 0], color=Colors.I_HAT, stroke_width=4)
        i_scaled_label = Tex(r"3\hat{\imath}", color=Colors.I_HAT, font_size=32)
        i_scaled_label.next_to(i_scaled, DOWN, buff=0.15)
        
        self.play(TransformFromCopy(self.lc_i_hat, i_scaled), run_time=1.5)
        self.play(Write(i_scaled_label))
        self.wait(1)
        
        # Step 2: Scale j-hat by -2 (flip and stretch)
        self.play(FadeOut(step1_text))
        step2_text = Text("y-coordinate (-2) flips and scales j-hat",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.J_HAT)
        step2_text.to_edge(UP, buff=0.5)
        self.play(Write(step2_text))
        
        # Animate j-hat scaling (flip and stretch)
        j_scaled = Vector([0, -2], color=Colors.J_HAT, stroke_width=4)
        j_scaled_label = Tex(r"-2\hat{\jmath}", color=Colors.J_HAT, font_size=32)
        j_scaled_label.next_to(j_scaled.get_end(), LEFT, buff=0.15)
        
        self.play(TransformFromCopy(self.lc_j_hat, j_scaled), run_time=1.5)
        self.play(Write(j_scaled_label))
        self.wait(1)
        
        # Step 3: Add them together
        self.play(FadeOut(step2_text))
        step3_text = Text("Add the scaled vectors together",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        step3_text.to_edge(UP, buff=0.5)
        self.play(Write(step3_text))
        
        # Move j_scaled to tip of i_scaled
        j_scaled_moved = Vector([0, -2], color=Colors.J_HAT, stroke_width=4)
        j_scaled_moved.shift([3, 0, 0])
        
        self.play(
            Transform(j_scaled, j_scaled_moved),
            j_scaled_label.animate.next_to([3, -1, 0], RIGHT, buff=0.15),
            run_time=1
        )
        
        # Draw result
        result_v = Vector([3, -2], color=Colors.RESULT, stroke_width=5)
        result_label = Tex(r"\begin{bmatrix} 3 \\ -2 \end{bmatrix} = 3\hat{\imath} + (-2)\hat{\jmath}",
                          color=Colors.RESULT, font_size=32)
        result_label.to_corner(UR, buff=0.8)
        
        self.play(GrowArrow(result_v), Write(result_label))
        self.wait(1.5)
        
        # This is a "linear combination"
        self.play(FadeOut(step3_text))
        lc_def = Text("This is called a 'linear combination'",
                     font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        lc_def.to_edge(UP, buff=0.5)
        self.play(Write(lc_def))
        self.wait(1)
        
        # General formula
        self.play(FadeOut(lc_def))
        formula = Tex(r"a\vec{v} + b\vec{w}", font_size=48, color=Colors.TEXT_PRIMARY)
        formula.to_edge(UP, buff=0.5)
        self.play(Write(formula))
        
        formula_desc = Text("Scale two vectors, then add them",
                           font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        formula_desc.next_to(formula, DOWN, buff=0.3)
        self.play(Write(formula_desc))
        self.wait(2)
        
        # Cleanup - also restore basis vectors visibility for span demo
        self.play(
            FadeOut(i_scaled), FadeOut(j_scaled), FadeOut(result_v),
            FadeOut(i_scaled_label), FadeOut(j_scaled_label), FadeOut(result_label),
            FadeOut(formula), FadeOut(formula_desc),
            self.lc_i_hat.animate.set_opacity(1),
            self.lc_j_hat.animate.set_opacity(1),
            self.lc_i_label.animate.set_opacity(1),
            self.lc_j_label.animate.set_opacity(1),
        )
    
    def _lincomb_span_2d(self):
        """Explain the span of vectors in 2D."""
        
        # What is span?
        span_def = Text("The 'span' is all vectors you can reach",
                       font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        span_def.to_edge(UP, buff=0.5)
        self.play(Write(span_def))
        
        span_def2 = Text("using linear combinations",
                        font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        span_def2.next_to(span_def, DOWN, buff=0.2)
        self.play(Write(span_def2))
        self.wait(1.5)
        
        # Interactive demonstration - vary scalars
        # First hide the basis vectors for cleaner visualization
        self.play(
            FadeOut(span_def), FadeOut(span_def2),
            FadeOut(self.lc_i_hat), FadeOut(self.lc_j_hat),
            FadeOut(self.lc_i_label), FadeOut(self.lc_j_label)
        )
        vary_text = Text("Vary the scalars a and b...",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        vary_text.to_edge(UP, buff=0.5)
        self.play(Write(vary_text))
        
        # Show many different linear combinations
        vectors_shown = VGroup()
        combinations = [
            (1, 0), (0, 1), (2, 1), (1, 2), (-1, 1), (1, -1),
            (-2, 1), (1.5, 2), (-1, -1), (2, -1.5), (0.5, 2.5), (-2, -2)
        ]
        
        for i, (a, b) in enumerate(combinations):
            v = Vector([a, b], color=Colors.ACCENT, stroke_width=2)
            v.set_opacity(0.6)
            self.play(GrowArrow(v), run_time=0.2)
            vectors_shown.add(v)
        
        self.wait(1)
        
        # The span fills the entire plane
        self.play(FadeOut(vary_text))
        fill_text = Text("With i-hat and j-hat, you can reach ANY vector!",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        fill_text.to_edge(UP, buff=0.5)
        self.play(Write(fill_text))
        
        span_result = Text("The span is the entire 2D plane",
                          font_size=Config.BODY_FONT_SIZE - 2, color=Colors.SUCCESS)
        span_result.next_to(fill_text, DOWN, buff=0.3)
        self.play(Write(span_result))
        self.wait(2)
        
        # What if vectors line up?
        self.play(FadeOut(fill_text), FadeOut(span_result), FadeOut(vectors_shown))
        
        lineup_text = Text("But what if the two vectors line up?",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        lineup_text.to_edge(UP, buff=0.5)
        self.play(Write(lineup_text))
        
        # Show two collinear vectors
        v1_collinear = Vector([2, 1], color=Colors.PRIMARY, stroke_width=5)
        v2_collinear = Vector([-1, -0.5], color=Colors.SECONDARY, stroke_width=5)
        
        v1_label = Tex(r"\vec{v}", color=Colors.PRIMARY, font_size=32)
        v1_label.next_to(v1_collinear.get_end(), UR, buff=0.1)
        v2_label = Tex(r"\vec{w}", color=Colors.SECONDARY, font_size=32)
        v2_label.next_to(v2_collinear.get_end(), DL, buff=0.1)
        
        self.play(GrowArrow(v1_collinear), Write(v1_label))
        self.play(GrowArrow(v2_collinear), Write(v2_label))
        self.wait(1)
        
        # Draw the line they span
        span_line = Line([-5, -2.5, 0], [5, 2.5, 0], color=Colors.ACCENT, stroke_width=2)
        
        self.play(FadeOut(lineup_text))
        line_text = Text("Their span is just a line!",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.ERROR)
        line_text.to_edge(UP, buff=0.5)
        self.play(Write(line_text), ShowCreation(span_line))
        self.wait(2)
        
        # Cleanup
        self.play(
            FadeOut(v1_collinear), FadeOut(v2_collinear),
            FadeOut(v1_label), FadeOut(v2_label),
            FadeOut(span_line), FadeOut(line_text)
        )
    
    def _lincomb_linear_dependence(self):
        """Explain linear independence vs dependence."""
        
        dep_title = Text("Linear Independence vs Dependence",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        dep_title.to_edge(UP, buff=0.5)
        self.play(Write(dep_title))
        self.wait(1)
        
        # Independent case
        self.play(FadeOut(dep_title))
        indep_text = Text("Linearly Independent: each vector adds a new dimension",
                         font_size=Config.BODY_FONT_SIZE - 2, color=Colors.SUCCESS)
        indep_text.to_edge(UP, buff=0.5)
        self.play(Write(indep_text))
        
        v1 = Vector([2, 1], color=Colors.PRIMARY, stroke_width=5)
        v2 = Vector([1, 2], color=Colors.SECONDARY, stroke_width=5)
        
        self.play(GrowArrow(v1), GrowArrow(v2))
        
        indep_span = Text("Span = entire 2D plane ✓",
                         font_size=24, color=Colors.SUCCESS)
        indep_span.to_corner(DL, buff=0.8)
        self.play(Write(indep_span))
        self.wait(1.5)
        
        self.play(FadeOut(v1), FadeOut(v2), FadeOut(indep_text), FadeOut(indep_span))
        
        # Dependent case
        dep_text = Text("Linearly Dependent: one vector is redundant",
                       font_size=Config.BODY_FONT_SIZE - 2, color=Colors.ERROR)
        dep_text.to_edge(UP, buff=0.5)
        self.play(Write(dep_text))
        
        v1_dep = Vector([2, 1], color=Colors.PRIMARY, stroke_width=5)
        v2_dep = Vector([4, 2], color=Colors.SECONDARY, stroke_width=5)  # = 2 * v1
        
        self.play(GrowArrow(v1_dep), GrowArrow(v2_dep))
        
        # Show that v2 = 2*v1
        relation = Tex(r"\vec{w} = 2\vec{v}", font_size=32, color=Colors.TEXT_SECONDARY)
        relation.to_corner(UL, buff=1).shift(DOWN * 1.5)
        self.play(Write(relation))
        
        dep_span = Text("Span = just a line ✗",
                       font_size=24, color=Colors.ERROR)
        dep_span.to_corner(DL, buff=0.8)
        self.play(Write(dep_span))
        self.wait(1.5)
        
        # One can be removed without reducing span
        self.play(FadeOut(dep_text))
        remove_text = Text("One vector can be removed without reducing the span",
                          font_size=Config.BODY_FONT_SIZE - 2, color=Colors.TEXT_SECONDARY)
        remove_text.to_edge(UP, buff=0.5)
        self.play(Write(remove_text))
        self.wait(2)
        
        # Technical definition of basis
        self.play(FadeOut(remove_text), FadeOut(v1_dep), FadeOut(v2_dep), 
                  FadeOut(relation), FadeOut(dep_span))
        
        basis_def = Text("Basis = linearly independent vectors that span the space",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        basis_def.to_edge(UP, buff=0.5)
        self.play(Write(basis_def))
        
        # Show i-hat and j-hat as THE basis
        i_hat_final = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        j_hat_final = Vector([0, 1], color=Colors.J_HAT, stroke_width=5)
        
        self.play(GrowArrow(i_hat_final), GrowArrow(j_hat_final))
        
        standard_text = Tex(r"\hat{\imath} \textrm{ and } \hat{\jmath} \textrm{ form the standard basis for } \mathbb{R}^2",
                           font_size=32, color=Colors.TEXT_PRIMARY)
        standard_text.to_edge(DOWN, buff=0.8)
        self.play(Write(standard_text))
        self.wait(2)
        
        # Final cleanup - clear everything from this chapter
        self.play(
            FadeOut(self.lc_plane),
            FadeOut(basis_def), FadeOut(i_hat_final), FadeOut(j_hat_final),
            FadeOut(standard_text)
        )
        
        # Brief pause before next chapter
        self.wait(0.5)
    
    # =========================================================================
    # CHAPTER 3: LINEAR TRANSFORMATIONS AND MATRICES
    # Based on 3Blue1Brown's Essence of Linear Algebra
    # =========================================================================
    
    def _chapter_transformations(self):
        self._create_chapter_header(3, "Linear Transformations")
        
        # --- PART 1: What is a transformation? ---
        self._transform_intro()
        
        # --- PART 2: What makes it "linear"? ---
        self._transform_linearity()
        
        # --- PART 3: Tracking basis vectors ---
        self._transform_basis_tracking()
        
        # --- PART 4: Matrix representation ---
        self._transform_matrix()
        
        # --- PART 5: Examples ---
        self._transform_examples()
    
    def _transform_intro(self):
        """Introduce transformations as functions that move vectors."""
        
        # Title concept
        intro_text = Text("A transformation takes vectors as input and output",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        intro_text.to_edge(UP, buff=0.5)
        self.play(Write(intro_text))
        self.wait(1)
        
        # Why "transformation"?
        self.play(FadeOut(intro_text))
        why_text = Text("'Transformation' suggests movement",
                       font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        why_text.to_edge(UP, buff=0.5)
        self.play(Write(why_text))
        
        # Show grid
        plane = self._create_grid(opacity=0.5)
        self.play(FadeIn(plane), run_time=0.8)
        
        # Show a vector moving
        v_start = Vector([2, 1], color=Colors.RESULT, stroke_width=5)
        self.play(GrowArrow(v_start))
        
        # Animate it "transforming" - create a new vector for the end state
        v_end = Vector([1, 2.5], color=Colors.RESULT, stroke_width=5)
        self.play(Transform(v_start, v_end), run_time=1.5)
        self.wait(1)
        
        # Think of every point moving
        self.play(FadeOut(why_text), FadeOut(v_start))
        points_text = Text("Imagine every point in space moving",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        points_text.to_edge(UP, buff=0.5)
        self.play(Write(points_text))
        
        # Add some dots to visualize
        dots = VGroup()
        for x in range(-4, 5):
            for y in range(-3, 4):
                if x != 0 or y != 0:
                    dot = Dot([x, y, 0], radius=0.06, color=Colors.TEXT_SECONDARY)
                    dot.set_opacity(0.5)
                    dots.add(dot)
        
        self.play(FadeIn(dots, lag_ratio=0.01))
        self.wait(1)
        
        # Store for next section
        self.tf_plane = plane
        self.tf_dots = dots
        self.play(FadeOut(points_text))
    
    def _transform_linearity(self):
        """Explain what makes a transformation 'linear'."""
        
        linear_text = Text("What makes a transformation 'linear'?",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        linear_text.to_edge(UP, buff=0.5)
        self.play(Write(linear_text))
        self.wait(1)
        
        # Two properties
        self.play(FadeOut(linear_text))
        prop1 = Text("1. All lines remain lines (no curves)",
                    font_size=Config.BODY_FONT_SIZE - 2, color=Colors.TEXT_SECONDARY)
        prop1.to_edge(UP, buff=0.5)
        self.play(Write(prop1))
        self.wait(1.5)
        
        prop2 = Text("2. The origin stays fixed",
                    font_size=Config.BODY_FONT_SIZE - 2, color=Colors.TEXT_SECONDARY)
        prop2.next_to(prop1, DOWN, buff=0.3)
        self.play(Write(prop2))
        self.wait(1.5)
        
        # Consequence
        self.play(FadeOut(prop1), FadeOut(prop2))
        consequence = Text("Grid lines remain parallel and evenly spaced",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        consequence.to_edge(UP, buff=0.5)
        self.play(Write(consequence))
        self.wait(2)
        
        self.play(FadeOut(consequence))
    
    def _transform_basis_tracking(self):
        """Show that tracking basis vectors tells you everything."""
        
        # Key insight
        key_text = Text("Key insight: just track where basis vectors land",
                       font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        key_text.to_edge(UP, buff=0.5)
        self.play(Write(key_text))
        
        # Add basis vectors
        i_hat = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        j_hat = Vector([0, 1], color=Colors.J_HAT, stroke_width=5)
        i_label = Tex(r"\hat{\imath}", color=Colors.I_HAT, font_size=32)
        i_label.next_to(i_hat.get_end(), DOWN, buff=0.1)
        j_label = Tex(r"\hat{\jmath}", color=Colors.J_HAT, font_size=32)
        j_label.next_to(j_hat.get_end(), LEFT, buff=0.1)
        
        self.play(GrowArrow(i_hat), GrowArrow(j_hat), Write(i_label), Write(j_label))
        self.wait(1)
        
        # Add an example vector
        v = Vector([-1, 2], color=Colors.RESULT, stroke_width=4)
        v_label = Tex(r"\vec{v} = -1\hat{\imath} + 2\hat{\jmath}", color=Colors.RESULT, font_size=28)
        v_label.to_corner(UL, buff=1).shift(DOWN * 1.5)
        
        self.play(GrowArrow(v), Write(v_label))
        self.wait(1)
        
        # Apply a transformation
        self.play(FadeOut(key_text))
        transform_text = Text("After transformation, v is still the same combination",
                             font_size=Config.BODY_FONT_SIZE - 2, color=Colors.TEXT_SECONDARY)
        transform_text.to_edge(UP, buff=0.5)
        self.play(Write(transform_text))
        
        # Define transformation: i-hat goes to [1, -2], j-hat goes to [3, 0]
        matrix = [[1, 3], [-2, 0]]
        
        # Prepare grid for transformation
        self.tf_plane.prepare_for_nonlinear_transform()
        
        # Apply transformation
        self.play(
            ApplyMatrix(matrix, self.tf_plane),
            ApplyMatrix(matrix, self.tf_dots),
            ApplyMatrix(matrix, i_hat),
            ApplyMatrix(matrix, j_hat),
            ApplyMatrix(matrix, v),
            i_label.animate.move_to([1.3, -2.3, 0]),
            j_label.animate.move_to([3.3, 0.3, 0]),
            run_time=3,
            rate_func=smooth
        )
        self.wait(1)
        
        # Show that v = -1*(new i-hat) + 2*(new j-hat)
        self.play(FadeOut(transform_text))
        result_text = Tex(r"\vec{v} = -1 \cdot \begin{bmatrix} 1 \\ -2 \end{bmatrix} + 2 \cdot \begin{bmatrix} 3 \\ 0 \end{bmatrix} = \begin{bmatrix} 5 \\ 2 \end{bmatrix}",
                         font_size=32, color=Colors.TEXT_PRIMARY)
        result_text.to_edge(UP, buff=0.5)
        self.play(Write(result_text))
        self.wait(2)
        
        # Cleanup and store
        self.play(
            FadeOut(self.tf_plane), FadeOut(self.tf_dots),
            FadeOut(i_hat), FadeOut(j_hat), FadeOut(v),
            FadeOut(i_label), FadeOut(j_label), FadeOut(v_label),
            FadeOut(result_text)
        )
    
    def _transform_matrix(self):
        """Show how a matrix encodes the transformation."""
        
        # A 2x2 matrix encodes the transformation
        matrix_text = Text("A 2x2 matrix encodes any linear transformation",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        matrix_text.to_edge(UP, buff=0.5)
        self.play(Write(matrix_text))
        
        # Show matrix structure
        matrix_structure = Tex(
            r"\begin{bmatrix} | & | \\ \hat{\imath}_{new} & \hat{\jmath}_{new} \\ | & | \end{bmatrix}",
            font_size=48, color=Colors.TEXT_PRIMARY
        )
        self.play(Write(matrix_structure))
        self.wait(1.5)
        
        # Concrete example
        self.play(FadeOut(matrix_text), FadeOut(matrix_structure))
        example_text = Text("Example: where do basis vectors land?",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        example_text.to_edge(UP, buff=0.5)
        self.play(Write(example_text))
        
        # Show grid
        plane = self._create_grid(opacity=0.4)
        plane.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane), run_time=0.5)
        
        i_hat = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        j_hat = Vector([0, 1], color=Colors.J_HAT, stroke_width=5)
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        
        # Show i-hat lands at [1, -2]
        i_lands = Tex(r"\hat{\imath} \to \begin{bmatrix} 1 \\ -2 \end{bmatrix}",
                     color=Colors.I_HAT, font_size=32)
        i_lands.to_corner(UL, buff=1).shift(DOWN * 1.5)
        
        # Show j-hat lands at [3, 0]
        j_lands = Tex(r"\hat{\jmath} \to \begin{bmatrix} 3 \\ 0 \end{bmatrix}",
                     color=Colors.J_HAT, font_size=32)
        j_lands.next_to(i_lands, DOWN, buff=0.3)
        
        self.play(Write(i_lands), Write(j_lands))
        
        # Apply transformation
        matrix = [[1, 3], [-2, 0]]
        self.play(
            ApplyMatrix(matrix, plane),
            ApplyMatrix(matrix, i_hat),
            ApplyMatrix(matrix, j_hat),
            run_time=2.5
        )
        self.wait(1)
        
        # Show the matrix
        self.play(FadeOut(example_text))
        matrix_result = Tex(
            r"\textrm{Matrix} = \begin{bmatrix} 1 & 3 \\ -2 & 0 \end{bmatrix}",
            font_size=40, color=Colors.ACCENT
        )
        matrix_result.to_edge(UP, buff=0.5)
        self.play(Write(matrix_result))
        self.wait(1)
        
        # Matrix-vector multiplication formula
        mult_formula = Tex(
            r"\begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = x \begin{bmatrix} a \\ c \end{bmatrix} + y \begin{bmatrix} b \\ d \end{bmatrix} = \begin{bmatrix} ax + by \\ cx + dy \end{bmatrix}",
            font_size=32, color=Colors.TEXT_PRIMARY
        )
        mult_formula.to_edge(DOWN, buff=0.8)
        self.play(Write(mult_formula))
        self.wait(2)
        
        # Cleanup
        self.play(
            FadeOut(plane), FadeOut(i_hat), FadeOut(j_hat),
            FadeOut(i_lands), FadeOut(j_lands),
            FadeOut(matrix_result), FadeOut(mult_formula)
        )
    
    def _transform_examples(self):
        """Show specific transformation examples: rotation and shear."""
        
        # --- ROTATION ---
        rotation_text = Text("Example: 90° counterclockwise rotation",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        rotation_text.to_edge(UP, buff=0.5)
        self.play(Write(rotation_text))
        
        plane = self._create_grid(opacity=0.4)
        plane.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane), run_time=0.5)
        
        i_hat = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        j_hat = Vector([0, 1], color=Colors.J_HAT, stroke_width=5)
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        
        # Show where they land
        rot_info = Tex(
            r"\hat{\imath} \to \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad \hat{\jmath} \to \begin{bmatrix} -1 \\ 0 \end{bmatrix}",
            font_size=32, color=Colors.TEXT_SECONDARY
        )
        rot_info.to_corner(UL, buff=1).shift(DOWN * 1.5)
        self.play(Write(rot_info))
        
        # Apply rotation
        rot_matrix = [[0, -1], [1, 0]]
        self.play(
            ApplyMatrix(rot_matrix, plane),
            ApplyMatrix(rot_matrix, i_hat),
            ApplyMatrix(rot_matrix, j_hat),
            run_time=2
        )
        
        rot_matrix_tex = Tex(
            r"\textrm{Rotation matrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}",
            font_size=32, color=Colors.ACCENT
        )
        rot_matrix_tex.to_edge(DOWN, buff=0.8)
        self.play(Write(rot_matrix_tex))
        self.wait(1.5)
        
        self.play(
            FadeOut(plane), FadeOut(i_hat), FadeOut(j_hat),
            FadeOut(rotation_text), FadeOut(rot_info), FadeOut(rot_matrix_tex)
        )
        
        # --- SHEAR ---
        shear_text = Text("Example: Shear transformation",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        shear_text.to_edge(UP, buff=0.5)
        self.play(Write(shear_text))
        
        plane2 = self._create_grid(opacity=0.4)
        plane2.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane2), run_time=0.5)
        
        i_hat2 = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        j_hat2 = Vector([0, 1], color=Colors.J_HAT, stroke_width=5)
        self.play(GrowArrow(i_hat2), GrowArrow(j_hat2))
        
        # Shear info
        shear_info = Tex(
            r"\hat{\imath} \to \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \hat{\jmath} \to \begin{bmatrix} 1 \\ 1 \end{bmatrix}",
            font_size=32, color=Colors.TEXT_SECONDARY
        )
        shear_info.to_corner(UL, buff=1).shift(DOWN * 1.5)
        self.play(Write(shear_info))
        
        # Apply shear
        shear_matrix = [[1, 1], [0, 1]]
        self.play(
            ApplyMatrix(shear_matrix, plane2),
            ApplyMatrix(shear_matrix, i_hat2),
            ApplyMatrix(shear_matrix, j_hat2),
            run_time=2
        )
        
        shear_matrix_tex = Tex(
            r"\textrm{Shear matrix} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}",
            font_size=32, color=Colors.ACCENT
        )
        shear_matrix_tex.to_edge(DOWN, buff=0.8)
        self.play(Write(shear_matrix_tex))
        self.wait(1.5)
        
        # Final insight
        self.play(FadeOut(shear_text), FadeOut(shear_info), FadeOut(shear_matrix_tex))
        insight = Text("Matrices are a language for describing transformations",
                      font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        insight.to_edge(UP, buff=0.5)
        self.play(Write(insight))
        self.wait(2)
        
        self.play(FadeOut(plane2), FadeOut(i_hat2), FadeOut(j_hat2), FadeOut(insight))
    
    # =========================================================================
    # CHAPTER 4: MATRIX MULTIPLICATION AS COMPOSITION
    # Based on 3Blue1Brown's Essence of Linear Algebra
    # =========================================================================
    
    def _chapter_matrix_multiplication(self):
        self._create_chapter_header(4, "Matrix Multiplication")
        
        # --- PART 1: Composition of Transformations ---
        self._matmul_composition_intro()
        
        # --- PART 2: Composition is Multiplication ---
        self._matmul_is_composition()
        
        # --- PART 3: Order Matters (Non-commutativity) ---
        self._matmul_order_matters()
        
        # --- PART 4: The Formula ---
        self._matmul_formula()
    
    def _matmul_composition_intro(self):
        """Introduce the idea of applying one transformation after another."""
        
        intro_text = Text("What happens when we apply two transformations?",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        intro_text.to_edge(UP, buff=0.5)
        self.play(Write(intro_text))
        self.wait(1)
        
        # Setup grid
        plane = self._create_grid(opacity=0.4)
        plane.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane), run_time=0.5)
        
        i_hat = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        j_hat = Vector([0, 1], color=Colors.J_HAT, stroke_width=5)
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        
        # First transformation: rotation 90°
        self.play(FadeOut(intro_text))
        rot_text = Text("First: Rotate 90° counterclockwise",
                       font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        rot_text.to_edge(UP, buff=0.5)
        self.play(Write(rot_text))
        
        rot_matrix = [[0, -1], [1, 0]]
        self.play(
            ApplyMatrix(rot_matrix, plane),
            ApplyMatrix(rot_matrix, i_hat),
            ApplyMatrix(rot_matrix, j_hat),
            run_time=2
        )
        self.wait(1)
        
        # Second transformation: shear
        self.play(FadeOut(rot_text))
        shear_text = Text("Then: Apply a shear",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.SECONDARY)
        shear_text.to_edge(UP, buff=0.5)
        self.play(Write(shear_text))
        
        shear_matrix = [[1, 1], [0, 1]]
        self.play(
            ApplyMatrix(shear_matrix, plane),
            ApplyMatrix(shear_matrix, i_hat),
            ApplyMatrix(shear_matrix, j_hat),
            run_time=2
        )
        self.wait(1)
        
        # The result is a new transformation
        self.play(FadeOut(shear_text))
        result_text = Text("The overall effect is a NEW transformation!",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        result_text.to_edge(UP, buff=0.5)
        self.play(Write(result_text))
        
        # This is called composition
        comp_text = Text("This is called 'composition'",
                        font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        comp_text.next_to(result_text, DOWN, buff=0.3)
        self.play(Write(comp_text))
        self.wait(2)
        
        self.play(FadeOut(plane), FadeOut(i_hat), FadeOut(j_hat),
                  FadeOut(result_text), FadeOut(comp_text))
    
    def _matmul_is_composition(self):
        """Show that matrix multiplication represents composition."""
        
        # The composition can be described by a single matrix
        single_text = Text("The composition can be described by ONE matrix",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        single_text.to_edge(UP, buff=0.5)
        self.play(Write(single_text))
        
        # Show the equation
        equation = Tex(
            r"\underbrace{\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}}_{\textrm{Shear}} \cdot \underbrace{\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}}_{\textrm{Rotation}} = \underbrace{\begin{bmatrix} 1 & -1 \\ 1 & 0 \end{bmatrix}}_{\textrm{Composition}}",
            font_size=36, color=Colors.TEXT_PRIMARY
        )
        self.play(Write(equation))
        self.wait(2)
        
        # Key insight: reading right to left
        self.play(FadeOut(single_text))
        order_text = Text("Read right to left! (like function notation)",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        order_text.to_edge(UP, buff=0.5)
        self.play(Write(order_text))
        
        # Show f(g(x)) analogy
        func_text = Tex(r"f(g(x)) \leftarrow \textrm{apply } g \textrm{ first, then } f",
                       font_size=32, color=Colors.TEXT_SECONDARY)
        func_text.next_to(equation, DOWN, buff=0.5)
        self.play(Write(func_text))
        self.wait(2)
        
        self.play(FadeOut(equation), FadeOut(order_text), FadeOut(func_text))
    
    def _matmul_order_matters(self):
        """Demonstrate that matrix multiplication is not commutative."""
        
        order_title = Text("Does order matter?",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        order_title.to_edge(UP, buff=0.5)
        self.play(Write(order_title))
        self.wait(1)
        
        # Setup two grids side by side with more spacing
        plane1 = self._create_grid(opacity=0.25)
        plane1.prepare_for_nonlinear_transform()
        plane1.scale(0.4).shift(LEFT * 3.5 + DOWN * 0.3)
        
        plane2 = self._create_grid(opacity=0.25)
        plane2.prepare_for_nonlinear_transform()
        plane2.scale(0.4).shift(RIGHT * 3.5 + DOWN * 0.3)
        
        self.play(FadeIn(plane1), FadeIn(plane2))
        
        # Labels positioned above grids with more buffer
        label1 = Text("Rotation then Shear", font_size=18, color=Colors.PRIMARY)
        label1.next_to(plane1, UP, buff=0.5)
        label2 = Text("Shear then Rotation", font_size=18, color=Colors.SECONDARY)
        label2.next_to(plane2, UP, buff=0.5)
        self.play(Write(label1), Write(label2))
        
        rot_matrix = [[0, -1], [1, 0]]
        shear_matrix = [[1, 1], [0, 1]]
        
        # Apply in different orders
        self.play(FadeOut(order_title))
        step1_text = Text("Apply first transformation...",
                         font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        step1_text.to_edge(UP, buff=0.3)
        self.play(Write(step1_text))
        
        self.play(
            ApplyMatrix(rot_matrix, plane1),
            ApplyMatrix(shear_matrix, plane2),
            run_time=2
        )
        self.wait(0.5)
        
        self.play(FadeOut(step1_text))
        step2_text = Text("Apply second transformation...",
                         font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        step2_text.to_edge(UP, buff=0.3)
        self.play(Write(step2_text))
        
        self.play(
            ApplyMatrix(shear_matrix, plane1),
            ApplyMatrix(rot_matrix, plane2),
            run_time=2
        )
        self.wait(1)
        
        # Result: they're different!
        self.play(FadeOut(step2_text))
        diff_text = Text("Different results! Order MATTERS!",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.ERROR)
        diff_text.to_edge(UP, buff=0.3)
        self.play(Write(diff_text))
        
        # Mathematical statement - position at bottom with clear space
        noncomm = Tex(r"AB \neq BA \textrm{ (in general)}",
                     font_size=36, color=Colors.ACCENT)
        noncomm.to_edge(DOWN, buff=0.5)
        self.play(Write(noncomm))
        self.wait(2)
        
        self.play(FadeOut(plane1), FadeOut(plane2), FadeOut(label1), FadeOut(label2),
                  FadeOut(diff_text), FadeOut(noncomm))
    
    def _matmul_formula(self):
        """Show the general formula for matrix multiplication."""
        
        formula_title = Text("The Matrix Multiplication Formula",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        formula_title.to_edge(UP, buff=0.5)
        self.play(Write(formula_title))
        
        # General formula
        formula = Tex(
            r"\begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} ae+bg & af+bh \\ ce+dg & cf+dh \end{bmatrix}",
            font_size=36, color=Colors.TEXT_PRIMARY
        )
        self.play(Write(formula))
        self.wait(1.5)
        
        # Explain the columns
        self.play(FadeOut(formula_title))
        col_text = Text("Each column = left matrix × corresponding column of right matrix",
                       font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        col_text.to_edge(UP, buff=0.5)
        self.play(Write(col_text))
        self.wait(2)
        
        # But remember the intuition!
        self.play(FadeOut(col_text))
        intuition = Text("But remember: it's just applying one transformation after another!",
                        font_size=Config.BODY_FONT_SIZE - 2, color=Colors.ACCENT)
        intuition.to_edge(UP, buff=0.5)
        self.play(Write(intuition))
        self.wait(2)
        
        # Associativity is trivial with this view
        self.play(FadeOut(intuition))
        assoc_text = Text("This view makes associativity obvious:",
                         font_size=Config.BODY_FONT_SIZE - 2, color=Colors.TEXT_SECONDARY)
        assoc_text.to_edge(UP, buff=0.5)
        self.play(Write(assoc_text))
        
        assoc_formula = Tex(r"(AB)C = A(BC)",
                           font_size=48, color=Colors.SUCCESS)
        assoc_formula.next_to(formula, DOWN, buff=0.5)
        self.play(Write(assoc_formula))
        
        assoc_explain = Text("Apply C, then B, then A — same either way!",
                            font_size=24, color=Colors.TEXT_SECONDARY)
        assoc_explain.next_to(assoc_formula, DOWN, buff=0.3)
        self.play(Write(assoc_explain))
        self.wait(2)
        
        self.play(FadeOut(formula), FadeOut(assoc_text), FadeOut(assoc_formula), FadeOut(assoc_explain))
    
    # =========================================================================
    # CHAPTER 5: THE DETERMINANT
    # Based on 3Blue1Brown's Essence of Linear Algebra
    # =========================================================================
    
    def _chapter_determinants(self):
        self._create_chapter_header(5, "The Determinant")
        
        # --- PART 1: Scaling Area ---
        self._det_scaling_area()
        
        # --- PART 2: What Determinant Tells Us ---
        self._det_meaning()
        
        # --- PART 3: Negative Determinant ---
        self._det_negative()
        
        # --- PART 4: Zero Determinant ---
        self._det_zero()
        
        # --- PART 5: The Formula ---
        self._det_formula()
    
    def _det_scaling_area(self):
        """Introduce determinant as area scaling factor."""
        
        intro = Text("How much does a transformation stretch or squish space?",
                    font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        intro.to_edge(UP, buff=0.5)
        self.play(Write(intro))
        
        # Setup grid with unit square
        plane = self._create_grid(opacity=0.3)
        plane.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane), run_time=0.5)
        
        # Unit square
        square = Square(side_length=1, color=Colors.RESULT, fill_opacity=0.5, stroke_width=3)
        square.move_to([0.5, 0.5, 0])
        
        area_label = Tex(r"\textrm{Area} = 1", font_size=28, color=Colors.RESULT)
        area_label.next_to(square, DOWN, buff=0.2)
        
        self.play(ShowCreation(square), Write(area_label))
        self.wait(1)
        
        # Apply scaling transformation
        self.play(FadeOut(intro))
        scale_text = Text("Apply a scaling transformation",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        scale_text.to_edge(UP, buff=0.5)
        self.play(Write(scale_text))
        
        matrix = [[3, 0], [0, 2]]
        self.play(
            ApplyMatrix(matrix, plane),
            ApplyMatrix(matrix, square),
            area_label.animate.move_to([1.5, -0.5, 0]),
            run_time=2
        )
        
        # New area
        new_area = Tex(r"\textrm{Area} = 6", font_size=28, color=Colors.ACCENT)
        new_area.next_to(square, DOWN, buff=0.2)
        self.play(Transform(area_label, new_area))
        self.wait(1)
        
        # This scaling factor is the determinant
        self.play(FadeOut(scale_text))
        det_intro = Text("This scaling factor is called the DETERMINANT",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        det_intro.to_edge(UP, buff=0.5)
        self.play(Write(det_intro))
        
        det_formula = Tex(r"\det\begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix} = 6",
                         font_size=36, color=Colors.TEXT_PRIMARY)
        det_formula.to_edge(DOWN, buff=0.8)
        self.play(Write(det_formula))
        self.wait(2)
        
        self.play(FadeOut(plane), FadeOut(square), FadeOut(area_label),
                  FadeOut(det_intro), FadeOut(det_formula))
    
    def _det_meaning(self):
        """Explain what the determinant tells us."""
        
        meaning_text = Text("The determinant tells you how areas scale",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        meaning_text.to_edge(UP, buff=0.5)
        self.play(Write(meaning_text))
        
        # Examples
        examples = VGroup(
            Tex(r"\det = 3 \Rightarrow \textrm{areas triple}", font_size=32),
            Tex(r"\det = 0.5 \Rightarrow \textrm{areas halve}", font_size=32),
            Tex(r"\det = 1 \Rightarrow \textrm{areas unchanged}", font_size=32),
        ).arrange(DOWN, buff=0.4)
        
        self.play(Write(examples))
        self.wait(2)
        
        # Key insight: works for ANY region
        self.play(FadeOut(meaning_text))
        any_region = Text("This works for ANY region, not just squares!",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        any_region.to_edge(UP, buff=0.5)
        self.play(Write(any_region))
        self.wait(2)
        
        self.play(FadeOut(examples), FadeOut(any_region))
    
    def _det_negative(self):
        """Explain negative determinants and orientation."""
        
        neg_title = Text("What about NEGATIVE determinants?",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        neg_title.to_edge(UP, buff=0.5)
        self.play(Write(neg_title))
        self.wait(1)
        
        # Show orientation
        plane = self._create_grid(opacity=0.3)
        plane.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane), run_time=0.5)
        
        i_hat = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        j_hat = Vector([0, 1], color=Colors.J_HAT, stroke_width=5)
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        
        # Note: j-hat is to the LEFT of i-hat
        self.play(FadeOut(neg_title))
        orient_text = Text("Notice: j-hat is to the LEFT of i-hat",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        orient_text.to_edge(UP, buff=0.5)
        self.play(Write(orient_text))
        self.wait(1)
        
        # Apply a flip transformation
        self.play(FadeOut(orient_text))
        flip_text = Text("Apply a transformation that 'flips' space",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.ERROR)
        flip_text.to_edge(UP, buff=0.5)
        self.play(Write(flip_text))
        
        flip_matrix = [[1, 2], [3, 4]]  # det = -2
        self.play(
            ApplyMatrix(flip_matrix, plane),
            ApplyMatrix(flip_matrix, i_hat),
            ApplyMatrix(flip_matrix, j_hat),
            run_time=2
        )
        self.wait(1)
        
        # Now j-hat is on the RIGHT
        self.play(FadeOut(flip_text))
        flipped_text = Text("Now j-hat is on the RIGHT of i-hat!",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        flipped_text.to_edge(UP, buff=0.5)
        self.play(Write(flipped_text))
        
        # This means negative determinant
        neg_det = Tex(r"\det < 0 \Rightarrow \textrm{orientation flipped}",
                     font_size=36, color=Colors.ERROR)
        neg_det.to_edge(DOWN, buff=0.8)
        self.play(Write(neg_det))
        self.wait(2)
        
        self.play(FadeOut(plane), FadeOut(i_hat), FadeOut(j_hat),
                  FadeOut(flipped_text), FadeOut(neg_det))
    
    def _det_zero(self):
        """Explain zero determinant - squishing to lower dimension."""
        
        zero_title = Text("What if the determinant is ZERO?",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        zero_title.to_edge(UP, buff=0.5)
        self.play(Write(zero_title))
        
        plane = self._create_grid(opacity=0.3)
        plane.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane), run_time=0.5)
        
        # Unit square
        square = Square(side_length=1, color=Colors.RESULT, fill_opacity=0.5, stroke_width=3)
        square.move_to([0.5, 0.5, 0])
        self.play(ShowCreation(square))
        
        # Apply singular transformation
        self.play(FadeOut(zero_title))
        singular_text = Text("Space gets squished onto a line!",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.ERROR)
        singular_text.to_edge(UP, buff=0.5)
        self.play(Write(singular_text))
        
        singular_matrix = [[1, 2], [0.5, 1]]  # det = 0
        self.play(
            ApplyMatrix(singular_matrix, plane),
            ApplyMatrix(singular_matrix, square),
            run_time=2
        )
        self.wait(1)
        
        # Area becomes zero
        zero_area = Tex(r"\det = 0 \Rightarrow \textrm{area becomes } 0",
                       font_size=36, color=Colors.ERROR)
        zero_area.to_edge(DOWN, buff=0.8)
        self.play(Write(zero_area))
        self.wait(1)
        
        # This means columns are linearly dependent
        self.play(FadeOut(singular_text))
        dependent = Text("This happens when columns are linearly dependent",
                        font_size=Config.BODY_FONT_SIZE - 2, color=Colors.TEXT_SECONDARY)
        dependent.to_edge(UP, buff=0.5)
        self.play(Write(dependent))
        self.wait(2)
        
        self.play(FadeOut(plane), FadeOut(square), FadeOut(zero_area), FadeOut(dependent))
    
    def _det_formula(self):
        """Show the 2D determinant formula."""
        
        formula_title = Text("The 2×2 Determinant Formula",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        formula_title.to_edge(UP, buff=0.5)
        self.play(Write(formula_title))
        
        formula = Tex(
            r"\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc",
            font_size=48, color=Colors.ACCENT
        )
        self.play(Write(formula))
        self.wait(1.5)
        
        # Intuition
        self.play(FadeOut(formula_title))
        intuition1 = Text("a·d = scaling in x and y directions",
                         font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        intuition1.to_edge(UP, buff=0.5)
        self.play(Write(intuition1))
        
        intuition2 = Text("b·c = 'diagonal stretching' correction",
                         font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        intuition2.next_to(intuition1, DOWN, buff=0.3)
        self.play(Write(intuition2))
        self.wait(2)
        
        # Property: det(AB) = det(A)det(B)
        self.play(FadeOut(intuition1), FadeOut(intuition2))
        prop_text = Text("Beautiful property:",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        prop_text.to_edge(UP, buff=0.5)
        self.play(Write(prop_text))
        
        prop_formula = Tex(r"\det(AB) = \det(A) \cdot \det(B)",
                          font_size=40, color=Colors.SUCCESS)
        prop_formula.next_to(formula, DOWN, buff=0.5)
        self.play(Write(prop_formula))
        
        prop_explain = Text("Applying two transformations multiplies their scaling factors!",
                           font_size=24, color=Colors.TEXT_SECONDARY)
        prop_explain.next_to(prop_formula, DOWN, buff=0.3)
        self.play(Write(prop_explain))
        self.wait(2)
        
        self.play(FadeOut(formula), FadeOut(prop_text), FadeOut(prop_formula), FadeOut(prop_explain))
    
    # =========================================================================
    # =========================================================================
    # CHAPTER 9: EIGENVECTORS AND EIGENVALUES
    # Based on 3Blue1Brown's Essence of Linear Algebra
    # =========================================================================
    
    def _chapter_eigenvectors(self):
        self._create_chapter_header(9, "Eigenvectors and Eigenvalues")
        
        # --- PART 1: The Core Idea - Vectors that stay on their span ---
        self._eigen_intro()
        
        # --- PART 2: A Concrete Example ---
        self._eigen_example()
        
        # --- PART 3: 3D Rotation Axis ---
        self._eigen_3d_rotation()
        
        # --- PART 4: The Computation ---
        self._eigen_computation()
        
        # --- PART 5: Special Cases ---
        self._eigen_special_cases()
        
        # --- PART 6: Diagonal Matrices & Eigenbasis ---
        self._eigen_diagonal()
    
    def _eigen_intro(self):
        """Introduce the core idea: vectors that stay on their span."""
        
        intro = Text("Consider what a transformation does to any vector...",
                    font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        intro.to_edge(UP, buff=0.5)
        self.play(Write(intro))
        
        # Setup grid
        plane = self._create_grid(opacity=0.3)
        plane.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane), run_time=0.5)
        
        # Show a random vector and its span (line through origin)
        v = Vector([1.5, 1], color=Colors.ACCENT, stroke_width=5)
        span_line = DashedLine([-4, -2.67, 0], [4, 2.67, 0], color=Colors.ACCENT, stroke_width=2)
        span_line.set_opacity(0.5)
        
        self.play(GrowArrow(v), ShowCreation(span_line))
        
        span_text = Text("The span of this vector (line through origin and tip)",
                        font_size=24, color=Colors.TEXT_SECONDARY)
        span_text.to_edge(DOWN, buff=0.8)
        self.play(Write(span_text))
        self.wait(1)
        
        # Apply transformation - vector gets knocked off its span
        self.play(FadeOut(intro), FadeOut(span_text))
        knocked_text = Text("Most vectors get knocked OFF their span",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.ERROR)
        knocked_text.to_edge(UP, buff=0.5)
        self.play(Write(knocked_text))
        
        matrix = [[3, 1], [0, 2]]
        self.play(
            ApplyMatrix(matrix, plane),
            ApplyMatrix(matrix, v),
            run_time=2
        )
        self.wait(1)
        
        # Cleanup
        self.play(FadeOut(plane), FadeOut(v), FadeOut(span_line), FadeOut(knocked_text))
        
        # But SOME vectors stay on their span
        special_text = Text("But SOME special vectors stay on their span!",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.SUCCESS)
        special_text.to_edge(UP, buff=0.5)
        self.play(Write(special_text))
        
        only_text = Text("They only get stretched or squished, not rotated",
                        font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        only_text.next_to(special_text, DOWN, buff=0.3)
        self.play(Write(only_text))
        
        # Definition
        def_text = Text("These are called EIGENVECTORS",
                       font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        def_text.next_to(only_text, DOWN, buff=0.5)
        self.play(Write(def_text))
        self.wait(2)
        
        self.play(FadeOut(special_text), FadeOut(only_text), FadeOut(def_text))
    
    def _eigen_example(self):
        """Show a concrete example with the matrix [[3,1],[0,2]]."""
        
        example_text = Text("Example: Let's find the eigenvectors",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        example_text.to_edge(UP, buff=0.5)
        self.play(Write(example_text))
        
        # Setup
        plane = self._create_grid(opacity=0.3)
        plane.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane), run_time=0.5)
        
        # Show matrix
        matrix_tex = Tex(r"A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}",
                        font_size=36, color=Colors.TEXT_PRIMARY)
        matrix_tex.to_corner(UL, buff=0.5)
        self.play(Write(matrix_tex))
        
        # Basis vectors
        i_hat = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        j_hat = Vector([0, 1], color=Colors.J_HAT, stroke_width=5)
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        
        # i-hat is an eigenvector! Show its span
        self.play(FadeOut(example_text))
        ihat_text = Text("i-hat stays on the x-axis (its span)!",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.I_HAT)
        ihat_text.to_edge(UP, buff=0.5)
        self.play(Write(ihat_text))
        
        x_axis_line = DashedLine([-6, 0, 0], [6, 0, 0], color=Colors.I_HAT, stroke_width=2)
        x_axis_line.set_opacity(0.5)
        self.play(ShowCreation(x_axis_line))
        
        # Apply transformation
        matrix = [[3, 1], [0, 2]]
        self.play(
            ApplyMatrix(matrix, plane),
            ApplyMatrix(matrix, i_hat),
            ApplyMatrix(matrix, j_hat),
            ApplyMatrix(matrix, x_axis_line),
            run_time=2.5
        )
        self.wait(1)
        
        # i-hat is stretched by 3
        eigenvalue_text = Tex(r"\hat{\imath} \textrm{ is stretched by } 3 \textrm{ (eigenvalue } \lambda = 3)",
                             font_size=28, color=Colors.I_HAT)
        eigenvalue_text.to_edge(DOWN, buff=0.8)
        self.play(Write(eigenvalue_text))
        self.wait(1.5)
        
        # Cleanup and show another eigenvector
        self.play(FadeOut(plane), FadeOut(i_hat), FadeOut(j_hat), 
                  FadeOut(x_axis_line), FadeOut(ihat_text), FadeOut(eigenvalue_text))
        
        # Second eigenvector: [-1, 1]
        plane2 = self._create_grid(opacity=0.3)
        plane2.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane2), run_time=0.5)
        
        sneaky_text = Text("There's another eigenvector: [-1, 1]",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.SECONDARY)
        sneaky_text.to_edge(UP, buff=0.5)
        self.play(Write(sneaky_text))
        
        v_eigen = Vector([-1, 1], color=Colors.SECONDARY, stroke_width=5)
        diag_line = DashedLine([4, -4, 0], [-4, 4, 0], color=Colors.SECONDARY, stroke_width=2)
        diag_line.set_opacity(0.5)
        
        self.play(GrowArrow(v_eigen), ShowCreation(diag_line))
        
        # Apply same transformation
        self.play(
            ApplyMatrix(matrix, plane2),
            ApplyMatrix(matrix, v_eigen),
            ApplyMatrix(matrix, diag_line),
            run_time=2.5
        )
        
        eigen2_text = Tex(r"\textrm{Stretched by } 2 \textrm{ (eigenvalue } \lambda = 2)",
                         font_size=28, color=Colors.SECONDARY)
        eigen2_text.to_edge(DOWN, buff=0.8)
        self.play(Write(eigen2_text))
        self.wait(1.5)
        
        # Summary
        self.play(FadeOut(sneaky_text), FadeOut(eigen2_text))
        summary = VGroup(
            Text("This transformation has two eigenvectors:", font_size=28, color=Colors.TEXT_PRIMARY),
            Tex(r"\textrm{x-axis vectors: } \lambda = 3", font_size=28, color=Colors.I_HAT),
            Tex(r"\textrm{diagonal vectors: } \lambda = 2", font_size=28, color=Colors.SECONDARY),
        ).arrange(DOWN, buff=0.3)
        summary.to_edge(UP, buff=0.5)
        self.play(Write(summary))
        self.wait(2)
        
        self.play(FadeOut(plane2), FadeOut(v_eigen), FadeOut(diag_line), 
                  FadeOut(matrix_tex), FadeOut(summary))
    
    def _eigen_3d_rotation(self):
        """Show why eigenvectors matter: 3D rotation axis."""
        
        why_text = Text("Why care about eigenvectors?",
                       font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        why_text.to_edge(UP, buff=0.5)
        self.play(Write(why_text))
        self.wait(1)
        
        self.play(FadeOut(why_text))
        rotation_text = Text("Consider a 3D rotation...",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.TEXT_SECONDARY)
        rotation_text.to_edge(UP, buff=0.5)
        self.play(Write(rotation_text))
        
        # Show that eigenvector = axis of rotation
        insight = Text("The eigenvector IS the axis of rotation!",
                      font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        insight.next_to(rotation_text, DOWN, buff=0.4)
        self.play(Write(insight))
        
        explanation = Text("(It stays on its span while everything else rotates around it)",
                          font_size=24, color=Colors.TEXT_SECONDARY)
        explanation.next_to(insight, DOWN, buff=0.3)
        self.play(Write(explanation))
        
        # Eigenvalue = 1 for rotation (no stretching)
        eigenval = Tex(r"\lambda = 1 \textrm{ (rotation doesn't stretch)}",
                      font_size=28, color=Colors.TEXT_SECONDARY)
        eigenval.next_to(explanation, DOWN, buff=0.5)
        self.play(Write(eigenval))
        self.wait(2)
        
        self.play(FadeOut(rotation_text), FadeOut(insight), 
                  FadeOut(explanation), FadeOut(eigenval))
        
        # General insight
        general = Text("Eigenvectors reveal what a transformation 'really does'",
                      font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        general.to_edge(UP, buff=0.5)
        self.play(Write(general))
        
        general2 = Text("Independent of your coordinate system choice",
                       font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        general2.next_to(general, DOWN, buff=0.3)
        self.play(Write(general2))
        self.wait(2)
        
        self.play(FadeOut(general), FadeOut(general2))
    
    def _eigen_computation(self):
        """Show how to compute eigenvalues and eigenvectors."""
        
        compute_title = Text("How to find eigenvalues",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        compute_title.to_edge(UP, buff=0.5)
        self.play(Write(compute_title))
        
        # The defining equation
        equation = Tex(r"A\vec{v} = \lambda\vec{v}",
                      font_size=48, color=Colors.TEXT_PRIMARY)
        self.play(Write(equation))
        self.wait(1)
        
        # Rewrite
        self.play(FadeOut(compute_title))
        rewrite_text = Text("Rewrite as a matrix equation...",
                           font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        rewrite_text.to_edge(UP, buff=0.5)
        self.play(Write(rewrite_text))
        
        step1 = Tex(r"A\vec{v} - \lambda I \vec{v} = \vec{0}",
                   font_size=40, color=Colors.TEXT_PRIMARY)
        step1.next_to(equation, DOWN, buff=0.5)
        self.play(Write(step1))
        self.wait(0.5)
        
        step2 = Tex(r"(A - \lambda I)\vec{v} = \vec{0}",
                   font_size=40, color=Colors.ACCENT)
        step2.next_to(step1, DOWN, buff=0.3)
        self.play(Write(step2))
        self.wait(1)
        
        # Key insight
        self.play(FadeOut(rewrite_text))
        key_text = Text("For a non-zero solution, the matrix must squish space!",
                       font_size=Config.BODY_FONT_SIZE - 2, color=Colors.PRIMARY)
        key_text.to_edge(UP, buff=0.5)
        self.play(Write(key_text))
        
        det_eq = Tex(r"\det(A - \lambda I) = 0",
                    font_size=48, color=Colors.SUCCESS)
        det_eq.next_to(step2, DOWN, buff=0.5)
        self.play(Write(det_eq))
        self.wait(1.5)
        
        # This is the characteristic polynomial
        self.play(FadeOut(key_text))
        char_text = Text("This is the 'characteristic polynomial'",
                        font_size=Config.BODY_FONT_SIZE - 2, color=Colors.TEXT_SECONDARY)
        char_text.to_edge(UP, buff=0.5)
        self.play(Write(char_text))
        self.wait(2)
        
        self.play(FadeOut(equation), FadeOut(step1), FadeOut(step2), 
                  FadeOut(det_eq), FadeOut(char_text))
        
        # Concrete example
        example_title = Text("Example calculation",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        example_title.to_edge(UP, buff=0.5)
        self.play(Write(example_title))
        
        matrix_ex = Tex(r"A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}",
                       font_size=36, color=Colors.TEXT_PRIMARY)
        self.play(Write(matrix_ex))
        
        det_calc = Tex(r"\det\begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) = 0",
                      font_size=32, color=Colors.TEXT_PRIMARY)
        det_calc.next_to(matrix_ex, DOWN, buff=0.5)
        self.play(Write(det_calc))
        self.wait(1)
        
        solutions = Tex(r"\lambda = 3 \textrm{ or } \lambda = 2",
                       font_size=40, color=Colors.SUCCESS)
        solutions.next_to(det_calc, DOWN, buff=0.4)
        self.play(Write(solutions))
        self.wait(2)
        
        self.play(FadeOut(example_title), FadeOut(matrix_ex), 
                  FadeOut(det_calc), FadeOut(solutions))
    
    def _eigen_special_cases(self):
        """Show special cases: rotation (no real eigenvalues), shear."""
        
        # Rotation - no real eigenvectors
        rot_title = Text("Special case: 90° Rotation",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        rot_title.to_edge(UP, buff=0.5)
        self.play(Write(rot_title))
        
        plane = self._create_grid(opacity=0.3)
        plane.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane), run_time=0.5)
        
        # Show a vector
        v = Vector([2, 1], color=Colors.ACCENT, stroke_width=5)
        self.play(GrowArrow(v))
        
        # Apply 90 degree rotation
        rot_matrix = [[0, -1], [1, 0]]
        self.play(
            ApplyMatrix(rot_matrix, plane),
            ApplyMatrix(rot_matrix, v),
            run_time=2
        )
        
        self.play(FadeOut(rot_title))
        no_eigen = Text("Every vector gets rotated off its span!",
                       font_size=Config.BODY_FONT_SIZE, color=Colors.ERROR)
        no_eigen.to_edge(UP, buff=0.5)
        self.play(Write(no_eigen))
        
        # Math explanation
        rot_calc = Tex(r"\det\begin{pmatrix} -\lambda & -1 \\ 1 & -\lambda \end{pmatrix} = \lambda^2 + 1 = 0",
                      font_size=32, color=Colors.TEXT_PRIMARY)
        rot_calc.to_edge(DOWN, buff=0.8)
        self.play(Write(rot_calc))
        
        no_real = Text("No real solutions! (only imaginary: ±i)",
                      font_size=24, color=Colors.ERROR)
        no_real.next_to(rot_calc, UP, buff=0.3)
        self.play(Write(no_real))
        self.wait(2)
        
        self.play(FadeOut(plane), FadeOut(v), FadeOut(no_eigen), 
                  FadeOut(rot_calc), FadeOut(no_real))
        
        # Shear - only one eigenvalue
        shear_title = Text("Special case: Shear",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        shear_title.to_edge(UP, buff=0.5)
        self.play(Write(shear_title))
        
        plane2 = self._create_grid(opacity=0.3)
        plane2.prepare_for_nonlinear_transform()
        self.play(FadeIn(plane2), run_time=0.5)
        
        # Show x-axis vectors are eigenvectors
        i_hat = Vector([1, 0], color=Colors.I_HAT, stroke_width=5)
        x_line = DashedLine([-6, 0, 0], [6, 0, 0], color=Colors.I_HAT, stroke_width=2)
        x_line.set_opacity(0.5)
        self.play(GrowArrow(i_hat), ShowCreation(x_line))
        
        # Apply shear
        shear_matrix = [[1, 1], [0, 1]]
        self.play(
            ApplyMatrix(shear_matrix, plane2),
            ApplyMatrix(shear_matrix, i_hat),
            ApplyMatrix(shear_matrix, x_line),
            run_time=2
        )
        
        self.play(FadeOut(shear_title))
        shear_result = Text("Only x-axis vectors stay on their span!",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.I_HAT)
        shear_result.to_edge(UP, buff=0.5)
        self.play(Write(shear_result))
        
        shear_eigen = Tex(r"\lambda = 1 \textrm{ (vectors fixed, not stretched)}",
                         font_size=28, color=Colors.TEXT_SECONDARY)
        shear_eigen.to_edge(DOWN, buff=0.8)
        self.play(Write(shear_eigen))
        self.wait(2)
        
        self.play(FadeOut(plane2), FadeOut(i_hat), FadeOut(x_line),
                  FadeOut(shear_result), FadeOut(shear_eigen))
    
    def _eigen_diagonal(self):
        """Show the power of diagonal matrices and eigenbasis."""
        
        diag_title = Text("The Power of Diagonal Matrices",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        diag_title.to_edge(UP, buff=0.5)
        self.play(Write(diag_title))
        
        # When basis vectors are eigenvectors
        basis_text = Text("When basis vectors ARE eigenvectors...",
                         font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        basis_text.next_to(diag_title, DOWN, buff=0.4)
        self.play(Write(basis_text))
        
        # Show diagonal matrix
        diag_matrix = Tex(r"\begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix}",
                         font_size=48, color=Colors.ACCENT)
        self.play(Write(diag_matrix))
        
        diag_explain = Text("Matrix is diagonal! Eigenvalues on the diagonal.",
                           font_size=24, color=Colors.TEXT_SECONDARY)
        diag_explain.next_to(diag_matrix, DOWN, buff=0.4)
        self.play(Write(diag_explain))
        self.wait(1.5)
        
        # Why this is useful
        self.play(FadeOut(diag_title), FadeOut(basis_text))
        power_text = Text("Computing powers becomes EASY!",
                         font_size=Config.BODY_FONT_SIZE, color=Colors.SUCCESS)
        power_text.to_edge(UP, buff=0.5)
        self.play(Write(power_text))
        
        power_formula = Tex(r"\begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}^{100} = \begin{bmatrix} 3^{100} & 0 \\ 0 & 2^{100} \end{bmatrix}",
                           font_size=36, color=Colors.TEXT_PRIMARY)
        power_formula.next_to(diag_matrix, DOWN, buff=0.8)
        self.play(Write(power_formula))
        self.wait(1.5)
        
        # Eigenbasis
        self.play(FadeOut(power_text), FadeOut(diag_explain))
        eigenbasis_text = Text("An 'eigenbasis' makes computations beautiful",
                              font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        eigenbasis_text.to_edge(UP, buff=0.5)
        self.play(Write(eigenbasis_text))
        
        strategy = Text("Strategy: Change to eigenbasis → compute → change back",
                       font_size=24, color=Colors.TEXT_SECONDARY)
        strategy.next_to(eigenbasis_text, DOWN, buff=0.3)
        self.play(Write(strategy))
        self.wait(2)
        
        self.play(FadeOut(diag_matrix), FadeOut(power_formula),
                  FadeOut(eigenbasis_text), FadeOut(strategy))
    
    # =========================================================================
    # CHAPTER 7: DOT PRODUCTS AND DUALITY
    # Based on 3Blue1Brown's Essence of Linear Algebra
    # =========================================================================
    
    def _chapter_dot_product(self):
        self._create_chapter_header(7, "Dot Products and Duality")
        
        # --- PART 1: Numerical Computation ---
        self._dot_numerical()
        
        # --- PART 2: Geometric Interpretation ---
        self._dot_geometric()
        
        # --- PART 3: Order Doesn't Matter ---
        self._dot_order()
        
        # --- PART 4: Connection to Linear Transformations ---
        self._dot_transformations()
        
        # --- PART 5: Duality ---
        self._dot_duality()
    
    def _dot_numerical(self):
        """Introduce the numerical computation of dot product."""
        
        intro = Text("The Dot Product: A numerical view",
                    font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        intro.to_edge(UP, buff=0.5)
        self.play(Write(intro))
        
        # Show the formula
        formula_text = Text("Pair up coordinates, multiply, and add:",
                           font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        formula_text.next_to(intro, DOWN, buff=0.4)
        self.play(Write(formula_text))
        
        # Example
        example = Tex(r"\begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot \begin{bmatrix} 3 \\ 4 \end{bmatrix} = 1 \cdot 3 + 2 \cdot 4 = 11",
                     font_size=40, color=Colors.TEXT_PRIMARY)
        self.play(Write(example))
        self.wait(1.5)
        
        # Another example
        self.play(FadeOut(intro), FadeOut(formula_text))
        example2_title = Text("Works for any dimension:",
                             font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        example2_title.to_edge(UP, buff=0.5)
        self.play(Write(example2_title))
        
        example2 = Tex(r"\begin{bmatrix} 6 \\ 2 \\ 8 \\ 3 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 8 \\ 5 \\ 3 \end{bmatrix} = 6 \cdot 1 + 2 \cdot 8 + 8 \cdot 5 + 3 \cdot 3 = 71",
                      font_size=32, color=Colors.TEXT_PRIMARY)
        example2.next_to(example, DOWN, buff=0.5)
        self.play(Write(example2))
        self.wait(2)
        
        self.play(FadeOut(example), FadeOut(example2), FadeOut(example2_title))
    
    def _dot_geometric(self):
        """Show the geometric interpretation - projection."""
        
        geo_title = Text("Geometric Interpretation: Projection",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        geo_title.to_edge(UP, buff=0.5)
        self.play(Write(geo_title))
        
        # Setup
        plane = self._create_grid(opacity=0.2)
        self.play(FadeIn(plane), run_time=0.5)
        
        # Two vectors
        v = Vector([3, 1], color=Colors.PRIMARY, stroke_width=5)
        w = Vector([1, 2], color=Colors.SECONDARY, stroke_width=5)
        
        v_label = Tex(r"\vec{v}", color=Colors.PRIMARY, font_size=32)
        v_label.next_to(v.get_end(), RIGHT, buff=0.1)
        w_label = Tex(r"\vec{w}", color=Colors.SECONDARY, font_size=32)
        w_label.next_to(w.get_end(), UP, buff=0.1)
        
        self.play(GrowArrow(v), Write(v_label))
        self.play(GrowArrow(w), Write(w_label))
        self.wait(1)
        
        # Show projection of w onto v
        self.play(FadeOut(geo_title))
        proj_text = Text("Project w onto the line through v",
                        font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        proj_text.to_edge(UP, buff=0.5)
        self.play(Write(proj_text))
        
        # Calculate projection
        v_vec = np.array([3, 1])
        w_vec = np.array([1, 2])
        proj_scalar = np.dot(w_vec, v_vec) / np.dot(v_vec, v_vec)
        proj_point = proj_scalar * v_vec
        
        # Dashed line from w to projection
        proj_line = DashedLine([w_vec[0], w_vec[1], 0], [proj_point[0], proj_point[1], 0],
                               color=Colors.ACCENT, stroke_width=2)
        proj_dot = Dot([proj_point[0], proj_point[1], 0], color=Colors.ACCENT, radius=0.1)
        
        self.play(ShowCreation(proj_line), GrowFromCenter(proj_dot))
        
        # Projection vector
        proj_vec = Vector([proj_point[0], proj_point[1]], color=Colors.ACCENT, stroke_width=4)
        proj_label = Text("projection", font_size=20, color=Colors.ACCENT)
        proj_label.next_to(proj_vec.get_center(), DOWN, buff=0.2)
        self.play(GrowArrow(proj_vec), Write(proj_label))
        self.wait(1)
        
        # Formula
        self.play(FadeOut(proj_text))
        formula = Tex(r"\vec{v} \cdot \vec{w} = |\vec{v}| \times (\textrm{length of projection})",
                     font_size=32, color=Colors.TEXT_PRIMARY)
        formula.to_edge(UP, buff=0.5)
        self.play(Write(formula))
        self.wait(2)
        
        # Sign tells alignment
        self.play(FadeOut(formula))
        sign_text = Text("The SIGN tells you about alignment:",
                        font_size=Config.BODY_FONT_SIZE - 2, color=Colors.ACCENT)
        sign_text.to_edge(UP, buff=0.5)
        self.play(Write(sign_text))
        
        signs = VGroup(
            Tex(r"\vec{v} \cdot \vec{w} > 0", font_size=28, color=Colors.SUCCESS),
            Text(" → similar direction", font_size=24, color=Colors.TEXT_SECONDARY),
        ).arrange(RIGHT, buff=0.2)
        signs2 = VGroup(
            Tex(r"\vec{v} \cdot \vec{w} = 0", font_size=28, color=Colors.PRIMARY),
            Text(" → perpendicular", font_size=24, color=Colors.TEXT_SECONDARY),
        ).arrange(RIGHT, buff=0.2)
        signs3 = VGroup(
            Tex(r"\vec{v} \cdot \vec{w} < 0", font_size=28, color=Colors.ERROR),
            Text(" → opposite direction", font_size=24, color=Colors.TEXT_SECONDARY),
        ).arrange(RIGHT, buff=0.2)
        
        all_signs = VGroup(signs, signs2, signs3).arrange(DOWN, buff=0.3)
        all_signs.to_edge(DOWN, buff=0.8)
        self.play(Write(all_signs))
        self.wait(2)
        
        self.play(FadeOut(plane), FadeOut(v), FadeOut(w), FadeOut(v_label), FadeOut(w_label),
                  FadeOut(proj_line), FadeOut(proj_dot), FadeOut(proj_vec), FadeOut(proj_label),
                  FadeOut(sign_text), FadeOut(all_signs))
    
    def _dot_order(self):
        """Show that order doesn't matter in dot product."""
        
        order_title = Text("Surprise: Order doesn't matter!",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        order_title.to_edge(UP, buff=0.5)
        self.play(Write(order_title))
        
        # Show both interpretations give same result
        equation = Tex(r"\vec{v} \cdot \vec{w} = \vec{w} \cdot \vec{v}",
                      font_size=48, color=Colors.ACCENT)
        self.play(Write(equation))
        self.wait(1)
        
        # Explanation
        self.play(FadeOut(order_title))
        explain = Text("Project v onto w, or w onto v — same result!",
                      font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        explain.to_edge(UP, buff=0.5)
        self.play(Write(explain))
        
        intuition = Text("Scaling one vector scales both interpretations equally",
                        font_size=24, color=Colors.TEXT_SECONDARY)
        intuition.next_to(equation, DOWN, buff=0.5)
        self.play(Write(intuition))
        self.wait(2)
        
        self.play(FadeOut(equation), FadeOut(explain), FadeOut(intuition))
    
    def _dot_transformations(self):
        """Connect dot products to 1D linear transformations."""
        
        trans_title = Text("The Deeper View: Linear Transformations",
                          font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        trans_title.to_edge(UP, buff=0.5)
        self.play(Write(trans_title))
        
        # 2D to 1D transformation
        trans_text = Text("Consider transformations from 2D to 1D (the number line):",
                         font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        trans_text.next_to(trans_title, DOWN, buff=0.4)
        self.play(Write(trans_text))
        
        # Show 1x2 matrix
        matrix = Tex(r"\begin{bmatrix} 2 & -1 \end{bmatrix}",
                    font_size=48, color=Colors.TEXT_PRIMARY)
        self.play(Write(matrix))
        self.wait(1)
        
        # Matrix-vector multiplication
        self.play(FadeOut(trans_title), FadeOut(trans_text))
        mult_text = Text("Multiplying this matrix by a vector...",
                        font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        mult_text.to_edge(UP, buff=0.5)
        self.play(Write(mult_text))
        
        mult_example = Tex(r"\begin{bmatrix} 2 & -1 \end{bmatrix} \begin{bmatrix} 3 \\ 2 \end{bmatrix} = 2 \cdot 3 + (-1) \cdot 2 = 4",
                          font_size=36, color=Colors.TEXT_PRIMARY)
        mult_example.next_to(matrix, DOWN, buff=0.5)
        self.play(Write(mult_example))
        self.wait(1)
        
        # This looks like a dot product!
        self.play(FadeOut(mult_text))
        insight = Text("This looks exactly like a dot product!",
                      font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        insight.to_edge(UP, buff=0.5)
        self.play(Write(insight))
        
        dot_compare = Tex(r"\begin{bmatrix} 2 \\ -1 \end{bmatrix} \cdot \begin{bmatrix} 3 \\ 2 \end{bmatrix} = 4",
                         font_size=36, color=Colors.ACCENT)
        dot_compare.next_to(mult_example, DOWN, buff=0.4)
        self.play(Write(dot_compare))
        self.wait(2)
        
        self.play(FadeOut(matrix), FadeOut(mult_example), FadeOut(insight), FadeOut(dot_compare))
    
    def _dot_duality(self):
        """Explain the concept of duality."""
        
        duality_title = Text("Duality: A Beautiful Correspondence",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        duality_title.to_edge(UP, buff=0.5)
        self.play(Write(duality_title))
        
        # The key insight
        insight1 = Text("Every 1×2 matrix corresponds to a 2D vector",
                       font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        insight1.next_to(duality_title, DOWN, buff=0.5)
        self.play(Write(insight1))
        
        correspondence = Tex(r"\begin{bmatrix} a & b \end{bmatrix} \leftrightarrow \begin{bmatrix} a \\ b \end{bmatrix}",
                            font_size=48, color=Colors.ACCENT)
        self.play(Write(correspondence))
        self.wait(1)
        
        # Applying transformation = taking dot product
        self.play(FadeOut(duality_title), FadeOut(insight1))
        equiv_text = Text("Applying the transformation = taking the dot product!",
                         font_size=Config.BODY_FONT_SIZE - 2, color=Colors.SUCCESS)
        equiv_text.to_edge(UP, buff=0.5)
        self.play(Write(equiv_text))
        
        # This is duality
        duality_def = Text("This correspondence is called 'duality'",
                          font_size=Config.BODY_FONT_SIZE - 2, color=Colors.ACCENT)
        duality_def.next_to(equiv_text, DOWN, buff=0.4)
        self.play(Write(duality_def))
        self.wait(1)
        
        # Deeper meaning
        self.play(FadeOut(equiv_text), FadeOut(duality_def))
        deep = Text("Vectors can be thought of as transformations in disguise!",
                   font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        deep.to_edge(UP, buff=0.5)
        self.play(Write(deep))
        
        final_thought = Text("A vector IS a linear transformation to 1D",
                            font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        final_thought.next_to(correspondence, DOWN, buff=0.5)
        self.play(Write(final_thought))
        self.wait(2)
        
        self.play(FadeOut(correspondence), FadeOut(deep), FadeOut(final_thought))
    
    # =========================================================================
    # CHAPTER 8: CROSS PRODUCTS
    # Based on 3Blue1Brown's Essence of Linear Algebra
    # =========================================================================
    
    def _chapter_cross_product(self):
        self._create_chapter_header(8, "Cross Products")
        
        # --- PART 1: 2D Cross Product (Area) ---
        self._cross_2d()
        
        # --- PART 2: Orientation and Sign ---
        self._cross_orientation()
        
        # --- PART 3: Connection to Determinant ---
        self._cross_determinant()
        
        # --- PART 4: 3D Cross Product ---
        self._cross_3d()
        
        # --- PART 5: Computing with Determinant Trick ---
        self._cross_computation()
    
    def _cross_2d(self):
        """Introduce the 2D cross product as area of parallelogram."""
        
        intro = Text("The 2D Cross Product: Area",
                    font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        intro.to_edge(UP, buff=0.5)
        self.play(Write(intro))
        
        # Setup
        plane = self._create_grid(opacity=0.2)
        self.play(FadeIn(plane), run_time=0.5)
        
        # Two vectors
        v = Vector([3, 1], color=Colors.PRIMARY, stroke_width=5)
        w = Vector([1, 2], color=Colors.SECONDARY, stroke_width=5)
        
        v_label = Tex(r"\vec{v}", color=Colors.PRIMARY, font_size=32)
        v_label.next_to(v.get_end(), DR, buff=0.1)
        w_label = Tex(r"\vec{w}", color=Colors.SECONDARY, font_size=32)
        w_label.next_to(w.get_end(), UL, buff=0.1)
        
        self.play(GrowArrow(v), Write(v_label))
        self.play(GrowArrow(w), Write(w_label))
        
        # Draw parallelogram
        self.play(FadeOut(intro))
        para_text = Text("The parallelogram they span:",
                        font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        para_text.to_edge(UP, buff=0.5)
        self.play(Write(para_text))
        
        # Parallelogram vertices: origin, v, v+w, w
        v_vec = np.array([3, 1, 0])
        w_vec = np.array([1, 2, 0])
        parallelogram = Polygon(
            [0, 0, 0], v_vec, v_vec + w_vec, w_vec,
            color=Colors.ACCENT, fill_opacity=0.3, stroke_width=2
        )
        self.play(ShowCreation(parallelogram))
        self.wait(1)
        
        # Cross product = area
        self.play(FadeOut(para_text))
        area_text = Text("Cross product = Area of this parallelogram",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        area_text.to_edge(UP, buff=0.5)
        self.play(Write(area_text))
        
        # Calculate: 3*2 - 1*1 = 5
        area_formula = Tex(r"\vec{v} \times \vec{w} = 3 \cdot 2 - 1 \cdot 1 = 5",
                          font_size=36, color=Colors.TEXT_PRIMARY)
        area_formula.to_edge(DOWN, buff=0.8)
        self.play(Write(area_formula))
        self.wait(2)
        
        self.play(FadeOut(plane), FadeOut(v), FadeOut(w), FadeOut(v_label), FadeOut(w_label),
                  FadeOut(parallelogram), FadeOut(area_text), FadeOut(area_formula))
    
    def _cross_orientation(self):
        """Explain how orientation affects the sign."""
        
        orient_title = Text("Orientation Matters!",
                           font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        orient_title.to_edge(UP, buff=0.5)
        self.play(Write(orient_title))
        
        # Setup
        plane = self._create_grid(opacity=0.2)
        self.play(FadeIn(plane), run_time=0.5)
        
        # Show positive orientation
        v1 = Vector([2, 0], color=Colors.PRIMARY, stroke_width=5)
        w1 = Vector([1, 2], color=Colors.SECONDARY, stroke_width=5)
        
        self.play(GrowArrow(v1), GrowArrow(w1))
        
        # w is counterclockwise from v → positive
        self.play(FadeOut(orient_title))
        pos_text = Text("w is counterclockwise from v → POSITIVE",
                       font_size=Config.BODY_FONT_SIZE - 4, color=Colors.SUCCESS)
        pos_text.to_edge(UP, buff=0.5)
        self.play(Write(pos_text))
        
        pos_result = Tex(r"\vec{v} \times \vec{w} > 0",
                        font_size=40, color=Colors.SUCCESS)
        pos_result.to_edge(DOWN, buff=0.8)
        self.play(Write(pos_result))
        self.wait(1.5)
        
        # Swap them
        self.play(FadeOut(v1), FadeOut(w1), FadeOut(pos_text), FadeOut(pos_result))
        
        neg_text = Text("Swap the vectors → NEGATIVE",
                       font_size=Config.BODY_FONT_SIZE - 4, color=Colors.ERROR)
        neg_text.to_edge(UP, buff=0.5)
        self.play(Write(neg_text))
        
        # Now w first, v second
        w2 = Vector([2, 0], color=Colors.SECONDARY, stroke_width=5)
        v2 = Vector([1, 2], color=Colors.PRIMARY, stroke_width=5)
        self.play(GrowArrow(w2), GrowArrow(v2))
        
        neg_result = Tex(r"\vec{w} \times \vec{v} = -(\vec{v} \times \vec{w})",
                        font_size=36, color=Colors.ERROR)
        neg_result.to_edge(DOWN, buff=0.8)
        self.play(Write(neg_result))
        self.wait(1.5)
        
        # Key formula
        self.play(FadeOut(neg_text))
        key = Text("Order matters! Swapping flips the sign.",
                  font_size=Config.BODY_FONT_SIZE, color=Colors.ACCENT)
        key.to_edge(UP, buff=0.5)
        self.play(Write(key))
        self.wait(2)
        
        self.play(FadeOut(plane), FadeOut(v2), FadeOut(w2), FadeOut(neg_result), FadeOut(key))
    
    def _cross_determinant(self):
        """Show connection to determinant."""
        
        det_title = Text("Connection to Determinant",
                        font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        det_title.to_edge(UP, buff=0.5)
        self.play(Write(det_title))
        
        # The formula
        formula = Tex(r"\vec{v} \times \vec{w} = \det\begin{pmatrix} v_1 & w_1 \\ v_2 & w_2 \end{pmatrix}",
                     font_size=40, color=Colors.TEXT_PRIMARY)
        self.play(Write(formula))
        self.wait(1)
        
        # Why? Because determinant measures area scaling!
        self.play(FadeOut(det_title))
        why = Text("Why? The determinant measures area scaling!",
                  font_size=Config.BODY_FONT_SIZE - 2, color=Colors.ACCENT)
        why.to_edge(UP, buff=0.5)
        self.play(Write(why))
        
        # Expand the formula
        expand = Tex(r"= v_1 \cdot w_2 - v_2 \cdot w_1",
                    font_size=36, color=Colors.TEXT_PRIMARY)
        expand.next_to(formula, DOWN, buff=0.4)
        self.play(Write(expand))
        self.wait(1)
        
        # Example
        example = Tex(r"\begin{bmatrix} 3 \\ 1 \end{bmatrix} \times \begin{bmatrix} 1 \\ 2 \end{bmatrix} = 3 \cdot 2 - 1 \cdot 1 = 5",
                     font_size=32, color=Colors.SECONDARY)
        example.next_to(expand, DOWN, buff=0.4)
        self.play(Write(example))
        self.wait(2)
        
        self.play(FadeOut(formula), FadeOut(why), FadeOut(expand), FadeOut(example))
    
    def _cross_3d(self):
        """Introduce the 3D cross product - output is a vector!"""
        
        intro_3d = Text("The 3D Cross Product: A Vector Output!",
                       font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        intro_3d.to_edge(UP, buff=0.5)
        self.play(Write(intro_3d))
        
        # Key difference
        diff = Text("In 3D, the cross product outputs a VECTOR, not a number",
                   font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        diff.next_to(intro_3d, DOWN, buff=0.4)
        self.play(Write(diff))
        
        # Properties
        self.play(FadeOut(intro_3d), FadeOut(diff))
        
        props_title = Text("Properties of the 3D cross product:",
                          font_size=Config.BODY_FONT_SIZE - 2, color=Colors.ACCENT)
        props_title.to_edge(UP, buff=0.5)
        self.play(Write(props_title))
        
        props = VGroup(
            Text("• Length = area of parallelogram", font_size=28, color=Colors.TEXT_PRIMARY),
            Text("• Direction = perpendicular to both vectors", font_size=28, color=Colors.TEXT_PRIMARY),
            Text("• Follows the right-hand rule", font_size=28, color=Colors.TEXT_PRIMARY),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        props.next_to(props_title, DOWN, buff=0.5)
        
        for prop in props:
            self.play(Write(prop))
            self.wait(0.5)
        
        self.wait(1)
        
        # Right hand rule explanation
        self.play(FadeOut(props_title), FadeOut(props))
        rhr = Text("Right-hand rule: point fingers along v, curl toward w",
                  font_size=Config.BODY_FONT_SIZE - 2, color=Colors.TEXT_SECONDARY)
        rhr.to_edge(UP, buff=0.5)
        self.play(Write(rhr))
        
        rhr2 = Text("Your thumb points in the direction of v × w",
                   font_size=Config.BODY_FONT_SIZE - 4, color=Colors.ACCENT)
        rhr2.next_to(rhr, DOWN, buff=0.3)
        self.play(Write(rhr2))
        self.wait(2)
        
        self.play(FadeOut(rhr), FadeOut(rhr2))
    
    def _cross_computation(self):
        """Show the determinant trick for computing 3D cross product."""
        
        compute_title = Text("Computing the 3D Cross Product",
                            font_size=Config.BODY_FONT_SIZE, color=Colors.PRIMARY)
        compute_title.to_edge(UP, buff=0.5)
        self.play(Write(compute_title))
        
        # The trick
        trick_text = Text("Use a 'strange' determinant with basis vectors:",
                         font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        trick_text.next_to(compute_title, DOWN, buff=0.4)
        self.play(Write(trick_text))
        
        # The formula
        det_formula = Tex(r"\vec{v} \times \vec{w} = \det\begin{pmatrix} \hat{\imath} & v_1 & w_1 \\ \hat{\jmath} & v_2 & w_2 \\ \hat{k} & v_3 & w_3 \end{pmatrix}",
                         font_size=36, color=Colors.TEXT_PRIMARY)
        self.play(Write(det_formula))
        self.wait(1.5)
        
        # Explain the weirdness
        self.play(FadeOut(compute_title), FadeOut(trick_text))
        weird = Text("Yes, putting vectors in a matrix is weird...",
                    font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        weird.to_edge(UP, buff=0.5)
        self.play(Write(weird))
        
        but = Text("But it gives you the right answer!",
                  font_size=Config.BODY_FONT_SIZE - 2, color=Colors.SUCCESS)
        but.next_to(weird, DOWN, buff=0.3)
        self.play(Write(but))
        self.wait(1)
        
        # Result formula
        self.play(FadeOut(weird), FadeOut(but))
        result_title = Text("Expanding the determinant gives:",
                           font_size=Config.BODY_FONT_SIZE - 4, color=Colors.TEXT_SECONDARY)
        result_title.to_edge(UP, buff=0.5)
        self.play(Write(result_title))
        
        result = Tex(r"\vec{v} \times \vec{w} = \begin{bmatrix} v_2 w_3 - v_3 w_2 \\ v_3 w_1 - v_1 w_3 \\ v_1 w_2 - v_2 w_1 \end{bmatrix}",
                    font_size=36, color=Colors.ACCENT)
        result.next_to(det_formula, DOWN, buff=0.5)
        self.play(Write(result))
        self.wait(1)
        
        # Duality insight
        self.play(FadeOut(result_title))
        duality = Text("Deep insight: This connects to duality!",
                      font_size=Config.BODY_FONT_SIZE - 2, color=Colors.PRIMARY)
        duality.to_edge(UP, buff=0.5)
        self.play(Write(duality))
        
        duality2 = Text("The cross product IS a linear transformation in disguise",
                       font_size=24, color=Colors.TEXT_SECONDARY)
        duality2.next_to(duality, DOWN, buff=0.3)
        self.play(Write(duality2))
        self.wait(2)
        
        self.play(FadeOut(det_formula), FadeOut(result), FadeOut(duality), FadeOut(duality2))
    
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
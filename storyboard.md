# Linear Algebra: From Geometry to Eigenfaces
**Session Length:** 60 Minutes
**Theme:** "The Geometry of Data"

## Timeline Overview
| Time | Section | Key Visual |
| :--- | :--- | :--- |
| 00:00 - 05:00 | **1. Introduction** | Montage: Robotics arm, Graphics grid, Face morphing |
| 05:00 - 15:00 | **2. Vectors & Spaces** | Arrows growing, adding, scaling in 2D grid |
| 15:00 - 25:00 | **3. The Matrix ($Ax$)** | The Grid transforming (shear, rotate) |
| 25:00 - 35:00 | **4. Span & Determinants** | 2D planes in 3D, Area squishing to 0 |
| 35:00 - 45:00 | **5. Eigenvectors** | The "fixed" vectors during a stretch |
| 45:00 - 55:00 | **6. PCA (Eigenfaces)** | High-D cloud of faces $\to$ Principal Axes |
| 55:00 - 60:00 | **7. Conclusion** | Reconstructing a face from components |

---

## Detailed Storyboard

### 1. Introduction (5 mins)
**Goal:** Hook the audience. Show that LA is not just tables of numbers, but the language of space and data.

*   **Scene 1.1: The World of Vectors**
    *   **Visual:** A 3D robotic arm moving (vectors defining joints). A video game character wireframe rotating. A grid of engine faces.
    *   **Narration:** "Linear Algebra is the language of space. It's how we move robots, render graphics, and even how computers 'see' faces."
    *   **Transition:** Zoom into a single pixel/point, fading everything else to white.

### 2. Vectors and Vector Spaces (10 mins)
**Goal:** Establish the geometric/numeric duality.

*   **Scene 2.1: The Arrow and the List**
    *   **Visual:** A yellow arrow $\vec{v}$ on a dark grid. Next to it, a column vector $\begin{bmatrix} 2 \\ 1 \end{bmatrix}$.
    *   **Action:** Move the arrow; the numbers update.
    *   **Narration:** "To a physicist, it's an arrow. To a computer scientist, it's a list. To us, it's both."

*   **Scene 2.2: Addition & Scaling**
    *   **Visual:** Two vectors $\vec{v}$ and $\vec{w}$.
    *   **Action:** Show "Tip-to-Tail" addition forming a parallelogram. Show scaling $2\vec{v}$ stretching the arrow.
    *   **Key Equation:** $c \cdot \vec{v}$

### 3. What $Ax$ Really Means (10 mins)
**Goal:** The core insight. Matrices are functions.

*   **Scene 3.1: The Grid Transformation**
    *   **Visual:** A standard square grid with basis vectors $\hat{i}$ (green) and $\hat{j}$ (red).
    *   **Action:** The *entire grid* morphs. Parallel lines remain parallel. Origin stays fixed.
    *   **Narration:** "A matrix isn't just a table of numbers. It's a function that transforms space."

*   **Scene 3.2: Tracking Basis Vectors**
    *   **Visual:** Focus on $\hat{i}$ landing at $[1, 0]$ and $\hat{j}$ landing at $[1, 1]$ (Shear).
    *   **Action:** Highlight the columns of the matrix $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$.
    *   **Key Insight:** "The columns of $A$ tell you where $\hat{i}$ and $\hat{j}$ land."

### 4. Linear Combinations, Span, & Determinants (10 mins)
**Goal:** Visualizing "space" and "collapsing".

*   **Scene 4.1: Span**
    *   **Visual:** Two vectors in 3D.
    *   **Action:** Show them generating a 2D plane (a sheet) cutting through space. Add a 3rd vector *on* the plane (dependent) vs *off* the plane (independent).

*   **Scene 4.2: The Determinant**
    *   **Visual:** The unit square (area 1).
    *   **Action:** Apply a transformation. The square stretches to a parallelogram.
    *   **Narration:** "The determinant is just the scaling factor of the area."
    *   **Action:** Squish the grid into a line. Show $\det(A) = 0$.

### 5. Eigenvectors & Eigenvalues (10 mins)
**Goal:** Finding the "axes" of the transformation.

*   **Scene 5.1: The Knocked-Off Vector**
    *   **Visual:** Apply a transformation. Most vectors get knocked off their span (they rotate).
    *   **Action:** Highlight one specific vector that *only* stretches.
    *   **Narration:** "This vector is special. It doesn't rotate. It is an *Eigenvector*."
    *   **Key Equation:** $A\vec{v} = \lambda \vec{v}$

### 6. PCA & Eigenfaces (10 mins)
**Goal:** Applied LA. High-dimensional vectors.

*   **Scene 6.1: Faces as Vectors**
    *   **Visual:** An image of a face ($62 \times 47$ pixels). Unroll it into a long vector of length 2914.
    *   **Narration:** "A face is just a point in a 2914-dimensional space."

*   **Scene 6.2: The Cloud**
    *   **Visual:** A 3D scatter plot (representing high-D). Each point is a face.
    *   **Action:** Show the "Mean Face" in the center. Center the data (move cloud to origin).

*   **Scene 6.3: Finding the Axes (Eigenfaces)**
    *   **Visual:** Rotate axes to align with the widest spread of the data.
    *   **Action:** Reveal the first "Principal Component". It looks like a ghostly face (Eigenface).
    *   **Narration:** "The eigenvectors of the covariance matrix are the 'ingredients' of faces."

### 7. Conclusion (5 mins)
**Goal:** Synthesis.

*   **Scene 7.1: Reconstruction**
    *   **Visual:** Start with the Mean Face. Add $w_1 \cdot (\text{Eigenface } 1)$. Add $w_2 \cdot (\text{Eigenface } 2)$.
    *   **Action:** The blurry face sharpens into a specific person.
    *   **Narration:** "We can describe a complex human face with just a few numbers. That is the power of Linear Algebra."

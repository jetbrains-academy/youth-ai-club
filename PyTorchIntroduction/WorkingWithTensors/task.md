In this task you have to implement short functions using pytorch. Implement all functions with only torch-vectorized operations without `for` cycles or `numpy`.

1. [`count_means`](course://PyTorchIntroduction/WorkingWithTensors/task.py:6): given `torch.tensor x` count means for each row and for each column and return results as a tuple
2. [`first_row_and_column`](course://PyTorchIntroduction/WorkingWithTensors/task.py:10): given `torch.tensor x` return first row and first column of x as a tuple
3. [`create_chessboard`](course://PyTorchIntroduction/WorkingWithTensors/task.py:14): given `num_rows` - number of rows and `num_cols` - number of columns return a `torch.tensor` of size `[num_rows x num_cols]` filled with $0$ and $1$ in chessboard manner. Top-left element (intersection of first row and first column) should contain $0$.
4. [`create_arithmetic_progressions`](course://PyTorchIntroduction/WorkingWithTensors/task.py:18): given `num_rows` - number of rows and `num_cols` - number of columns return a `torch.tensor` of size `[num_rows x num_cols]` where $i$-th row (starting $i$ from 1) contains arithmetic progression starting with $i$ with step $i$
5. [`create_arithmetic_progressions`](course://PyTorchIntroduction/WorkingWithTensors/task.py:22): given `torch.tensor x` return a `torch.tensor` obtained by flattening all dimensions except first. If $x$ has shape $[d_1 \times d_2 \ldots d_k]$ then the result should has shape $[d_1 \times d_2 \cdot \ldots \cdot d_k]$

[//]: # ()
[//]: # (This is a task description file.)

[//]: # (Its content will be displayed to a learner)

[//]: # (in the **Task Description** window.)

[//]: # ()
[//]: # (It supports both Markdown and HTML.)

[//]: # (To toggle the format, you can rename **task.md**)

[//]: # (to **task.html**, or vice versa.)

[//]: # (The default task description format can be changed)

[//]: # (in **Preferences | Tools | Education**,)

[//]: # (but this will not affect any existing task description files.)

[//]: # ()
[//]: # (The following features are available in)

[//]: # (**task.md/task.html** which are specific to the JetBrains Academy plugin:)

[//]: # ()
[//]: # (- Hints can be added anywhere in the task text.)

[//]: # (  Type "hint" and press Tab.)

[//]: # (  Hints should be added to an empty line in the task text.)

[//]: # (  In hints you can use both HTML and Markdown.)

[//]: # (<div class="hint">)

[//]: # ()
[//]: # (Text of your hint)

[//]: # ()
[//]: # (</div>)

[//]: # ()
[//]: # (- You may need to refer your learners to a particular lesson,)

[//]: # (task, or file. To achieve this, you can use the in-course links.)

[//]: # (Specify the path using the `[link_text]&#40;course://lesson1/task1/file1&#41;` format.)

[//]: # ()
[//]: # (- You can insert shortcuts in the task description.)

[//]: # (While **task.html/task.md** is open, right-click anywhere)

[//]: # (on the **Editor** tab and choose the **Insert shortcut** option)

[//]: # (from the context menu.)

[//]: # (For example: &shortcut:FileStructurePopup;.)

[//]: # ()
[//]: # (- Insert the &percnt;`IDE_NAME`&percnt; macro,)

[//]: # (which will be replaced by the actual IDE name.)

[//]: # (For example, **%IDE_NAME%**.)

[//]: # ()
[//]: # (- Insert PSI elements, by using links like)

[//]: # (`[element_description]&#40;psi_element://link.to.element&#41;`.)

[//]: # (To get such a link, right-click the class or method)

[//]: # (and select **Copy Reference**.)

[//]: # (Then press &shortcut:EditorPaste; to insert the link where appropriate.)

[//]: # (For example, a [link to the "contains" method]&#40;psi_element://java.lang.String#contains&#41;.)

[//]: # ()
[//]: # (- You can add link to file using **full path** like this:)

[//]: # (  `[file_link]&#40;file://lesson1/task1/file.txt&#41;`.)
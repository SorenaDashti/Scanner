# Python Automatic Document Scanner 📄

This repository contains a Python script (`document_scanner.py`) that uses OpenCV to automatically detect the corners of a document within an image, perform a perspective transform to obtain a top-down "scanned" view, and save the resulting image. It employs a multi-layered approach with varying parameter strictness and fallback mechanisms for increased robustness across different image conditions.

A test script (`test_scanner.py`) is included to demonstrate how to use the main scanning module.

## Features ✨

*   **Automatic Document Detection:** Identifies the main document boundaries in an image.
*   **Perspective Correction:** Applies a four-point perspective transform to rectify the document image.
*   **Multi-Layered Detection Strategy:** Uses several tiers of detection parameters (Strict, Balanced, Loose, Very Loose) to find the best candidate quadrilateral, increasing the chance of success.
*   **Fallback Mechanism:** If the tiered search fails, it attempts to find the document using the largest contour's approximation or minimum bounding rectangle.
*   **Preprocessing Options:** Includes standard grayscale/blurring and CLAHE for contrast enhancement before contour detection.
*   **Corner Refinement:** Optionally refines the detected corners using `cv2.cornerSubPix` for potentially higher accuracy.
*   **Configurable Parameters:** Offers numerous parameters within the `scan_document` function to fine-tune the detection process (area ratios, solidity, aspect ratios, thresholds, etc.).
*   **Debug Mode:** Can save intermediate processing steps (images and logs) for analysis and debugging.
*   **Language:** Code comments are primarily in Persian (فارسی).

## Requirements 🛠️

*   Python 3.x
*   OpenCV (`opencv-python`)
*   NumPy (`numpy`)
*   Matplotlib (`matplotlib`) - *Required for the `test_scanner.py` display*

## Installation ⚙️

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SorenaDashti/Scanner.git
    cd  Scanner-main
    ```
2.  **Install the required libraries:**
    ```bash
    pip install opencv-python numpy matplotlib
    ```

## Usage 🚀

There are two main ways to use the code:

**1. Using the Test Script (`test_scanner.py`)**

This is the easiest way to test the scanner on a single image.

*   **Modify `test_scanner.py`:**
    *   Change the `input_image_path` variable to the full path of the image you want to scan.
      ```python
      # Example (use raw string r"" or forward slashes /)
      input_image_path = r"C:\path\to\your\document.jpg"
      # or
      input_image_path = "test_images/my_doc.png"
      ```
    *   Optionally, change `output_image_path` to set where the scanned image will be saved.
    *   Set `save_debug = True` or `False` to enable/disable saving debug steps.
*   **Run the script:**
    ```bash
    python test_scanner.py
    ```
*   The script will process the image, print logs to the console, save the output (if specified), potentially save debug files (in a folder like `scan_results_corrected_layers/debug_steps`), and display the final scanned image using Matplotlib.

**2. Using `document_scanner.py` as a Module**

You can import the `scan_document` function into your own Python projects.

*   Make sure `document_scanner.py` is in your Python path or the same directory as your script.
*   Import and use the function:

    ```python
    import document_scanner
    import cv2
    import os

    # --- Configuration ---
    image_to_scan = "path/to/your/image.jpg"
    output_directory = "my_scan_results"
    enable_debug = True

    # --- Check if image exists ---
    if not os.path.exists(image_to_scan):
        print(f"Error: Image not found at {image_to_scan}")
    else:
        # --- Call the scanner ---
        # You can override default parameters here
        scanned_doc = document_scanner.scan_document(
            image_path=image_to_scan,
            save_debug_steps=enable_debug,
            output_dir_base=output_directory,
            interp_height=1000, # Example: Change resize height
            strict_solidity=0.92 # Example: Make strict tier stricter
            # ... add other parameters as needed
        )

        # --- Process the result ---
        if scanned_doc is not None:
            print("Document scanned successfully!")
            # Save or display the result
            output_path = os.path.join(output_directory, "scanned", f"{os.path.basename(image_to_scan)}_scanned.jpg")
            os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure output dir exists
            cv2.imwrite(output_path, scanned_doc)
            print(f"Saved result to: {output_path}")

            # Optional: Display result
            # cv2.imshow("Scanned Document", cv2.resize(scanned_doc, (600, 800))) # Resize for display
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            print("Failed to scan the document.")

    ```

## Parameters 🔧

The `scan_document` function in `document_scanner.py` has numerous parameters to control the scanning process. Some key ones include:

*   `image_path`: Path to the input image.
*   `save_debug_steps`: Boolean, if `True`, saves intermediate images and logs.
*   `output_dir_base`: Base directory for saving results and debug files.
*   `interp_height`: Height to resize the image to for processing (affects speed and detail).
*   `bilateral_*`, `gaussian_*`, `clahe_*`: Parameters for preprocessing filters.
*   `adaptive_thresh_*`, `morph_*`, `canny_*`, `dilation_*`: Parameters for contour detection methods.
*   `num_contours_to_check`, `epsilon_range`, `epsilon_steps`: Control how contours are approximated.
*   `strict_*`, `balanced_*`, `loose_*`, `very_loose_*`: Define the criteria (area ratio, solidity, aspect ratio, angle tolerance) for each detection tier.
*   `fallback_*`: Parameters for the final fallback detection method.
*   `refine_corner_active`, `refine_corner_window_size`, `refine_gray_blur_ksize`: Control corner refinement.

Refer to the function signature and comments within `document_scanner.py` for a full list and default values.

## Output 📁

*   **Scanned Image:** The primary output is the perspective-corrected image, typically saved in a `scanned` subfolder within the `output_dir_base`.
*   **Debug Files (Optional):** If `save_debug_steps=True`, a `debug_steps` subfolder is created within `output_dir_base`. It contains:
    *   Intermediate images from various processing stages (resized, grayscale, thresholded, contours found, refined corners, etc.), prefixed with numbers to indicate order.
    *   A `_log.txt` file containing the console output messages for that specific image run.

## How It Works (High Level) ⚙️

1.  **Load & Resize:** Reads the input image and resizes it to a consistent height (`interp_height`) for processing efficiency.
2.  **Preprocess:** Converts the image to grayscale and applies filtering (Bilateral/Gaussian Blur, optionally CLAHE) to reduce noise and enhance edges.
3.  **Contour Detection:** Uses two main strategies on the preprocessed images:
    *   Adaptive Thresholding + Morphology (Closing).
    *   Canny Edge Detection + Dilation + Morphology (Closing).
    *   Finds contours using both `RETR_LIST` and `RETR_EXTERNAL`.
4.  **Multi-Tiered Candidate Search:**
    *   Iterates through contours from different sources (standard/CLAHE preprocess, list/external retrieval).
    *   For each source, applies multiple tiers of checks (`Strict` -> `Balanced` -> `Loose` -> `Very Loose`) using `find_document_quad_candidates`.
    *   Each tier looks for 4-sided, convex polygons matching specific criteria for area, solidity, aspect ratio, and (optionally) corner angles.
    *   The first successful candidate found from the prioritized sources/tiers is selected.
5.  **Fallback Search:** If no candidate is found in the tiers, it attempts a fallback:
    *   Finds the largest contour satisfying basic area/shape criteria.
    *   Tries `approxPolyDP` on this contour.
    *   If that fails, tries finding the `minAreaRect`.
6.  **Corner Refinement (Optional):** If a candidate is found and refinement is active, `cv2.cornerSubPix` is used on a slightly blurred grayscale image to potentially improve corner location accuracy.
7.  **Perspective Transformation:** The final selected corners (refined or not) are scaled back to the original image dimensions. `cv2.getPerspectiveTransform` and `cv2.warpPerspective` are used to get the top-down view.
8.  **Save Output:** The final warped image is saved. Debug steps are saved if enabled.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License 📝

[Specify Your License Here - e.g., MIT License]

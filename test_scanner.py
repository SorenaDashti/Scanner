# test_scanner.py

import cv2
import matplotlib.pyplot as plt
import document_scanner # وارد کردن ماژول اسکنر
import os

# --- تنظیمات ---
# آدرس تصویر ورودی (حتما r را قبل از آدرس قرار دهید یا از / به جای \ استفاده کنید)
# input_image_path = r"C:\Users\Sorena\Downloads\Telegram Desktop\1.jpg"
# یا
input_image_path = "C:/Users/Sorena/Downloads/Telegram Desktop/14.jpg"


# آدرس اختیاری برای ذخیره تصویر خروجی
output_image_path = "scanned_1.jpg"

# آیا مراحل دیباگ ذخیره شوند؟
save_debug = True

# --- اجرای پردازش ---
if __name__ == "__main__":

    print(f"[*] Starting document scan for: {input_image_path}")

    # بررسی وجود فایل ورودی
    if not os.path.exists(input_image_path):
        print(f"[ERROR] Input file not found at: {input_image_path}")
    else:
        try:
            # فراخوانی تابع اصلی از ماژول
            scanned_image = document_scanner.scan_document(
                image_path=input_image_path,
                save_debug_steps=save_debug,
                # --- می‌توانید پارامترهای دیگر را هم اینجا تغییر دهید ---
                # interp_height=500,
                # adaptive_thresh_block_size=25,
                # adaptive_thresh_C=11,
                # min_area_factor=0.1,
            )

            # --- نمایش و ذخیره نتیجه ---
            if scanned_image is not None:
                print("[*] Scan successful!")

                # ذخیره تصویر نهایی
                if output_image_path:
                    try:
                        cv2.imwrite(output_image_path, scanned_image)
                        print(f"[*] Result saved to: {output_image_path}")
                        # اگر می‌خواهید پوشه حاوی فایل خروجی باز شود (فقط ویندوز):
                        # try:
                        #     os.startfile(os.path.dirname(os.path.abspath(output_image_path)))
                        # except AttributeError: # os.startfile فقط در ویندوز موجود است
                        #     print("[INFO] Cannot automatically open output folder on this OS.")

                    except Exception as e:
                        print(f"[ERROR] Could not save output image to {output_image_path}: {e}")

                # نمایش تصویر نهایی با Matplotlib
                plt.figure(figsize=(8, 10))
                 # تبدیل از BGR (OpenCV) به RGB (Matplotlib)
                plt.imshow(cv2.cvtColor(scanned_image, cv2.COLOR_BGR2RGB))
                plt.title(f'Scanned Document: {os.path.basename(input_image_path)}')
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            else:
                print("[*] Scan failed. No output generated.")

        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred during the test script execution: {e}")
            import traceback
            traceback.print_exc()
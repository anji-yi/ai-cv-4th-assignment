from recode import *


def augment(img):
    augmented_img = crop_image(img, random.uniform(0.5,1))
    augmented_img = random_light_color(augmented_img, 50)
    augmented_img = gamma_correction(augmented_img, random.uniform(0,2))
    augmented_img = rotation(augmented_img, random.randint(0,90))
    augmented_img = perspective_transformation(augmented_img, 200)
    return augmented_img

if __name__ == "__main__":

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(proj_dir), "assets")
    filepath = os.path.join(assets_dir, "lenna.jpg")

    # Read image
    img = read_image(filepath)

    augmented_img = augment(img)
    show_image(augmented_img)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

    
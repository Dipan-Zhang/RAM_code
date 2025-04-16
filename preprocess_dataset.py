import os
import numpy as np
from vision.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image
from PIL import Image
import matplotlib.pyplot as plt
import ipdb
import cv2
"script for labelling motion trajection, generating new dataset and save results into pkl file"

def interactive_trajectory_labeling(img):
    """
    Allow users to interactively label points on an image by clicking.
    
    Args:
        img: The input image to label points on
        
    Returns:
        List of (x, y) coordinates of labeled points
    """
    points = []
    
    fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
    ax.imshow(img)
    ax.set_title("Click to add points. Press 'Enter' when done, 'Backspace' to remove last point")
    
    scatter = ax.scatter([], [], c='red', marker='o', s=10)
    line, = ax.plot([], [], c='red', linestyle='-', linewidth=1)
    
    def update_plot():
        if points:
            x_coords, y_coords = zip(*points) if len(points) > 0 else ([], [])
            scatter.set_offsets(points)
            line.set_data(x_coords, y_coords)
        else:
            scatter.set_offsets([])
            line.set_data([], [])
        fig.canvas.draw()
    
    def on_click(event):
        if event.inaxes != ax:
            return
        points.append((event.xdata, event.ydata))
        print(f"Point added: ({event.xdata:.1f}, {event.ydata:.1f})")
        update_plot()
    
    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)
        elif event.key == 'backspace' and points:
            removed = points.pop()
            print(f"Removed point: ({removed[0]:.1f}, {removed[1]:.1f})")
            update_plot()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    plt.show()
    
    return points

def draw_trajectory(img, points, color='red', marker='o', size=10):
    """
    Draw points representing a trajectory on an image
    
    Args:
        img: The input image
        points: List of (x, y) points to draw
        color: Color of the points
        marker: Marker style
        size: Size of markers
    """
    fig = plt.figure(dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Display the image
    ax.imshow(img)
    
    # Extract x and y coordinates
    if len(points) > 0:
        x_coords, y_coords = zip(*points)
        ax.scatter(x_coords, y_coords, c=color, marker=marker, s=size)
        
        # Connect points with lines to show trajectory
        ax.plot(x_coords, y_coords, c=color, linestyle='-', linewidth=1)
    
    return fig


grounded_dino_model, sam_predictor = prepare_gsam_model(device='cuda')

img_dir = '/home/stud/zanr/code/RAM_code/assets/data/customize/close_the_slide_cabinet/raw_imgs'
save_dir = '/home/stud/zanr/code/RAM_code/assets/data/customize/close_the_slide_cabinet/vis'
task_name = img_dir.split('/')[-2]
obj_name = ' '.join(task_name.split('_')[-2:])
print(obj_name)
prompt = f'a whole picture of {obj_name}'

imgs = []
masked_imgs = []
masks = []
trajs=[]
names = []
img_fns = sorted(os.listdir(img_dir))
for img_fn in img_fns:
    img_path = os.path.join(img_dir, img_fn)
    img = cv2.imread(img_path, -1)[...,[2,1,0]]  # BGR to RGB
    
    # Resize image to fit within 640x480 while maintaining aspect ratio
    h, w = img.shape[:2]
    ratio = min(640/w, 480/h)
    new_size = (int(w * ratio), int(h * ratio))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    # Create a black canvas of 640x480
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Place the resized image in the center of the canvas
    x_offset = (640 - new_size[0]) // 2
    y_offset = (480 - new_size[1]) // 2
    canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = img
    
    img = canvas

    # get mask
    tgt_masks = inference_one_image(img, grounded_dino_model, sam_predictor,\
                            box_threshold=0.5, text_threshold=0.35, text_prompt=prompt,\
                            device="cuda").cpu().numpy() # you can set point_prompt to traj[0]
    tgt_mask = (tgt_masks[0][0] > 0).astype(np.uint8)
    mask = np.repeat(tgt_masks[0,0][:, :, np.newaxis], 3, axis=2).astype(np.uint8) # align with RAM implementation

    # Apply mask to the image
    masked_img = img.copy()
    masked_img[tgt_mask == 0] = 255  # Set pixels to 0 where mask is 0

    # draw 2d trajectory
    # Let the user label trajectory points
    print("Please label the trajectory points on the image. Press Enter when done.")
    trajectory_points = interactive_trajectory_labeling(masked_img)
    print(f"Labeled {len(trajectory_points)} points: {trajectory_points}")



    # Draw the final trajectory on the masked image
    fig = draw_trajectory(masked_img, trajectory_points)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save with higher DPI and quality settings
    fig_save_fn = os.path.join(save_dir, f"trajectory_{img_fn}")
    fig.savefig(fig_save_fn, dpi=300, bbox_inches='tight', pad_inches=0, 
                format='png', transparent=False)
    plt.close(fig)  # Close the figure to free memory
    
    print(f"Saved trajectory image to {fig_save_fn}")

    # append results to list

    imgs.append(img)
    masked_imgs.append(masked_img)
    masks.append(mask)
    trajs.append(trajectory_points)
    names.append(img_fn)

# save result to pkl
import pickle
data = {
    'img': imgs,
    'masked_img': masked_imgs,
    'mask': masks,
    'traj': trajs,
    'name': names
}
with open(os.path.join(save_dir, f'{task_name}_new.pkl'), 'wb') as f:
    pickle.dump(data, f)
print(f"Saved data to {os.path.join(save_dir, f'{task_name}_new.pkl')}")


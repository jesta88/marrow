/* Fly camera: WASD/EQ move, right-mouse-drag to look. Produces a Vulkan view-projection. */
#ifndef DEMO_CAMERA_H
#define DEMO_CAMERA_H

#include "linalg.h"

typedef struct GLFWwindow GLFWwindow;

typedef struct {
    vec3  pos;
    float yaw, pitch;     /* radians; yaw around +Y, pitch around camera-right */
    float move_speed;     /* metres/second                                    */
    float look_speed;     /* radians/pixel                                    */
    /* mouse-look drag state */
    int   looking;
    double last_x, last_y;
} Camera;

void  camera_init(Camera *c, vec3 pos, float yaw, float pitch);
vec3  camera_forward(const Camera *c);
/* Advance from GLFW input over dt seconds. */
void  camera_update(Camera *c, GLFWwindow *win, float dt);
/* world->clip for the given aspect ratio. */
mat4  camera_view_proj(const Camera *c, float aspect);

#endif /* DEMO_CAMERA_H */

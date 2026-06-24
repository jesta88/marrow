#include "camera.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define DEMO_PI 3.14159265358979323846f

void camera_init(Camera *c, vec3 pos, float yaw, float pitch) {
    c->pos = pos;
    c->yaw = yaw;
    c->pitch = pitch;
    c->move_speed = 6.0f;
    c->look_speed = 0.0030f;
    c->looking = 0;
    c->last_x = c->last_y = 0.0;
}

vec3 camera_forward(const Camera *c) {
    float cp = cosf(c->pitch), sp = sinf(c->pitch);
    float cy = cosf(c->yaw), sy = sinf(c->yaw);
    /* yaw=0,pitch=0 looks down -Z */
    return v3_normalize(v3(sy * cp, sp, -cy * cp));
}

void camera_update(Camera *c, GLFWwindow *win, float dt) {
    /* mouse-look while the right button is held */
    int rmb = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    double mx, my;
    glfwGetCursorPos(win, &mx, &my);
    if (rmb && !c->looking) {
        c->looking = 1;
        c->last_x = mx; c->last_y = my;
    } else if (!rmb) {
        c->looking = 0;
    }
    if (c->looking) {
        float dx = (float)(mx - c->last_x), dy = (float)(my - c->last_y);
        c->last_x = mx; c->last_y = my;
        c->yaw   += dx * c->look_speed;
        c->pitch -= dy * c->look_speed;
        const float lim = DEMO_PI * 0.49f;
        if (c->pitch >  lim) c->pitch =  lim;
        if (c->pitch < -lim) c->pitch = -lim;
    }

    vec3 fwd = camera_forward(c);
    vec3 right = v3_normalize(v3_cross(fwd, v3(0, 1, 0)));
    vec3 up = v3(0, 1, 0);

    float speed = c->move_speed * dt;
    if (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 4.0f;

    vec3 d = v3(0, 0, 0);
    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) d = v3_add(d, fwd);
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) d = v3_sub(d, fwd);
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) d = v3_add(d, right);
    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) d = v3_sub(d, right);
    if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) d = v3_add(d, up);
    if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) d = v3_sub(d, up);
    if (v3_dot(d, d) > 0.0f) c->pos = v3_add(c->pos, v3_scale(v3_normalize(d), speed));
}

mat4 camera_view_proj(const Camera *c, float aspect) {
    vec3 fwd = camera_forward(c);
    mat4 view = mat4_look_at(c->pos, v3_add(c->pos, fwd), v3(0, 1, 0));
    mat4 proj = mat4_perspective_vk(60.0f * (DEMO_PI / 180.0f), aspect, 0.05f, 500.0f);
    return mat4_mul(proj, view);
}

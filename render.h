#ifndef OPENGL_LIBRARY_H
#define OPENGL_LIBRARY_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"


struct CameraLoc {
    float* loc;
};

struct Vertices {
    float* xyz;
    unsigned int size;
};

struct Faces {
    int* idxs;
    unsigned int size;
};

struct DepthBuffer {
    float* buffer;
};


struct Camera {
    float focal_length;
    unsigned int width;
    unsigned int height;
    float near;
    float far;
    float cx;
    float cy;
};

int renderOFF(Vertices* vertices, Faces* faces, DepthBuffer* depthBuffer, Camera* camera_init, CameraLoc* cameraLoc);

#endif //OPENGL_LIBRARY_H

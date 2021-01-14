#include "render.h"
#include <iostream>
#include <unistd.h>

const char* vertexShaderFBO =
        "#version 330\n"
        "layout(location = 0)  in vec3 vp;"
        "uniform mat4 camera;"
        "uniform mat4 projection;"
        "void main() {"
        "  gl_Position = camera * projection * vec4(vp.x, vp.y, vp.z, 1.0);"
        "}";

const char* fragmentShaderFBO =
        "#version 330\n"
        "layout(location = 0) out float frag_depth;"
        "void main() {"
        "  frag_depth = gl_FragCoord.z;"
        "}";

const char* vertexShader =
        "#version 330\n"
        "layout(location = 0)  in vec3 vp;"
        "layout(location = 1) in vec2 vertexUV;"
        "out vec2 UV;"
        "uniform mat4 camera;"
        "uniform mat4 projection;"
        "void main() {"
        "  gl_Position =  camera * projection * vec4(vp.x, vp.y, vp.z, 1.0);"
        "  UV = vertexUV;"
        "}";

const char* fragmentShader =
        "#version 330\n"
        "out vec4 frag_colour;"
        "uniform sampler2D depthSampler;"
        "in vec2 UV;"
        "void main() {"
        "  float depthVal = texture(depthSampler, UV).r;"
        "  frag_colour = vec4(vec3(depthVal), 1);"
        "}";

float linearizeDepth(float depth, float near, float far)
{
    float z = depth * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void errorCallback(int error, const char* description)
{
    std::cout << "Error :: " << error << ", Description :: " << description << std::endl;
}

void processKeyBoardInput(GLFWwindow* glfWwindow)
{
    if(glfwGetKey(glfWwindow, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(glfWwindow, true);
    }

}

GLuint compileShader(GLenum shaderType, const char* shader, const char* shaderName)
{
    GLuint p = glCreateShader(shaderType);
    glShaderSource(p, 1, &shader, NULL);
    glCompileShader(p);

    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(p, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(p, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::" << shaderName << "::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return p;
}

GLuint createShader(const char* vertexShader, const char* fragmentShader)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShader, "VERTEX");
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShader, "FRAGMENT");

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vs);
    glAttachShader(shaderProgram, fs);
    glLinkProgram(shaderProgram);

    // check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return shaderProgram;
}

GLuint setVAO()
{
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    return vao;
}

GLuint setVBO(const float *data, unsigned int size, GLuint vertexAttribute)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    glEnableVertexAttribArray(vertexAttribute);
    glVertexAttribPointer(vertexAttribute, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    // unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

GLuint setDepthTexture(int size_x, int size_y)
{
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, size_x, size_y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

    return texture;
}

void handleWindowResize(GLFWwindow* glfWwindow, int width, int height)
{
    std::cout << "Window is being resized :: " << "width, " << width << " height, " << height << std::endl;
    glViewport(0, 0, width, height);
}

void setCameraMatrix(float f, float w, float h, float cx, float cy, float near, float far, float* cam_pt)
{
    // refer https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
    *cam_pt++ = 2 * f / w; *cam_pt++= 0; *cam_pt++ = 0; *cam_pt++ = 0;
    *cam_pt++ = 0; *cam_pt++ = -2 * f / h; *cam_pt++ = 0; *cam_pt++ = 0;
    *cam_pt++ = (w - 2 * cx) / w; *cam_pt++ = (h - 2 * cy) / h; *cam_pt++ = (-far - near) / (far - near); *cam_pt++ = -1;
    *cam_pt++ = 0; *cam_pt++ = 0; *cam_pt++ = -2 * near * far / (far - near); *cam_pt = 0;
}


int renderOFF(Vertices* vertices, Faces* faces, DepthBuffer* depthBuffer, Camera* camera_init, CameraLoc* cameraLoc)
{
    //region gl_init
    glfwSetErrorCallback(errorCallback);
    GLuint status = glfwInit();

    if (!status)
    {
        std::cout << "glfwInit failed" << std::endl;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef ON_SCREEN
    GLuint displayWindow = GLFW_TRUE;
#else
    GLuint displayWindow = GLFW_FALSE;
#endif

    glfwWindowHint(GLFW_VISIBLE, displayWindow);

    GLFWwindow* glfWwindow = glfwCreateWindow(camera_init->width, camera_init->height, "LearnOpenGL", NULL, NULL);

#ifdef ON_SCREEN
    if (glfWwindow == NULL)
    {
        std::cout << "Failed to create a window :::" << std::endl;
        glfwTerminate();
        return -1;
    } else {
        std::cout << "Window created successfully" << std::endl;
    }
#endif

    glfwMakeContextCurrent(glfWwindow);

#ifdef ON_SCREEN
    glfwSetFramebufferSizeCallback(glfWwindow, handleWindowResize);
#endif

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    //endregion

    // set vertex array objects (vao)
    GLuint vao = setVAO();

    // set vertex buffer objects (vbo)
    // for vertex coordinates shader
    GLuint vbo = setVBO(vertices->xyz, sizeof(*vertices->xyz) * vertices->size, 0);

    // set vbo for indices
    GLuint ivbo;
    glGenBuffers(1, &ivbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ivbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(*faces->idxs) * faces->size, faces->idxs, GL_STATIC_DRAW);

    // create framebuffer shader
    GLuint shaderProgramFBO = createShader(vertexShaderFBO, fragmentShaderFBO);

#ifdef ON_SCREEN
    // create renderer shader
    GLuint shaderProgram = createShader(vertexShader, fragmentShader);
#endif

    //set texture
    GLuint depthTexture = setDepthTexture(camera_init->width, camera_init->height);

    // set framebuffer
    unsigned int fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // attach texture to framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "Failed to bind framebuffer" << std::endl;
        return -1;
    }
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

#ifdef ON_SCREEN
    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "depthSampler"), 0);
    glUseProgram(0);
#endif

    glm::mat4 camera = glm::mat4(1.0f);
    glm::mat4 projection = glm::mat4(1.0f);

    setCameraMatrix(camera_init->focal_length, (float) camera_init->width, (float) camera_init->height,
                    camera_init->cx, camera_init->cy, camera_init->near, camera_init->far, &camera[0][0]);

//    std::cout << "Camera Matrix" << std::endl;
//    std::cout << camera[0][0] << " " << camera[0][1] << " " << camera[0][2] << " " << camera[0][3] << std::endl;
//    std::cout << camera[1][0] << " " << camera[1][1] << " " << camera[1][2] << " " << camera[1][3] << std::endl;
//    std::cout << camera[2][0] << " " << camera[2][1] << " " << camera[2][2] << " " << camera[2][3] << std::endl;
//    std::cout << camera[3][0] << " " << camera[3][1] << " " << camera[3][2] << " " << camera[3][3] << std::endl;

//    std::cout << "LookAt Matrix" << std::endl;
//    std::cout << projection[0][0] << " " << projection[0][1] << " " << projection[0][2] << " " << projection[0][3] << std::endl;
//    std::cout << projection[1][0] << " " << projection[1][1] << " " << projection[1][2] << " " << projection[1][3] << std::endl;
//    std::cout << projection[2][0] << " " << projection[2][1] << " " << projection[2][2] << " " << projection[2][3] << std::endl;
//    std::cout << projection[3][0] << " " << projection[3][1] << " " << projection[3][2] << " " << projection[3][3] << std::endl;

#ifdef ON_SCREEN
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "camera"), 1, GL_FALSE, glm::value_ptr(camera));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUseProgram(0);
#endif

    glUseProgram(shaderProgramFBO);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgramFBO, "camera"), 1, GL_FALSE, glm::value_ptr(camera));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgramFBO, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUseProgram(0);


    int depth_out_idx = 0;

    // bind to custom framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    //glDepthFunc(GL_LESS);

    glClearColor(0, 0, 0, 1); // clear screen to black
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgramFBO);
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, faces->size, GL_UNSIGNED_INT, 0);

    GLfloat* depth2D = new GLfloat[ camera_init->width * camera_init->height ];
    glReadPixels(0, 0, camera_init->width, camera_init->height, GL_DEPTH_COMPONENT, GL_FLOAT, depth2D);

    for (int i = 0; i < camera_init->height; i++) {
        for (int j = 0; j < camera_init->width; j++) {
            if (depth2D[(i * camera_init->width) + j] < 1) {
                depthBuffer->buffer[depth_out_idx++] =  linearizeDepth(depth2D[(i * camera_init->width) + j], camera_init->near, camera_init->far);
            }
            else {
                depthBuffer->buffer[depth_out_idx++] =  0;
            }
        }
    }
    delete[] depth2D;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);

#ifdef ON_SCREEN
    while(!glfwWindowShouldClose(glfWwindow)) {
        // default framebuffer
        glClearColor(0, 0, 0, 1); // clear screen to black
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        processKeyBoardInput(glfWwindow);

        //glDisable(GL_DEPTH_TEST);
        glUseProgram(shaderProgram);
        glBindVertexArray(vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        glDrawElements(GL_TRIANGLES, faces->size, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(glfWwindow);
        glfwPollEvents();
        //sleep(1);
   }
#endif
    depth_out_idx = 0;
    glUseProgram(0);


    glDeleteFramebuffers(1, &fbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    glfwTerminate();
    return 0;
}


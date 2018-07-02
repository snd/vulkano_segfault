#version 450

// this is basically an identity shader which doesn't modify the vertex positions

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}

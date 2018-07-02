#version 450

// descriptor set 0
layout(set = 0, binding = 0) uniform Autocorrelation {
    float data[480];
} autocorrelation;

layout(location = 0) out vec4 fragColor;

layout(push_constant) uniform PushConstants {
    float time;
    float width;
    float height;
} push_constants;

void main() {
    vec2 position = gl_FragCoord.xy / vec2(push_constants.width, push_constants.height);

    // goes from 0.5 to 1 and back to 0.5 every PI (3.14...) seconds
    float animated = 0.5 + 0.5 * sin(push_constants.time);

    vec3 color = vec3(0., 0., 0.);

    // becomes redder from left to right
    color.r = position.x;

    // becomes greener from top to bottom
    color.g = position.y;
    // color.g = animated;

    // goes from 0.5 to 1 and back to 0.5 every PI (3.14...) seconds
    color.b = animated;

    fragColor.rgb = color.rgb;
}

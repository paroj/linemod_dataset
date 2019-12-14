#version 120

attribute vec4 position;
uniform mat4 MVP;

uniform vec3 min;
uniform vec3 extent;

varying vec3 color;

void main()
{
    color = (position.xyz - min)/extent;
    gl_Position = MVP * position; // needs w for proper perspective correction
}

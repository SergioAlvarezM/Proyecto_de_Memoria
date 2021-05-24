/*
*     This program is free software: you can redistribute it and/or modify
*     it under the terms of the GNU General Public License as published by
*     the Free Software Foundation, either version 3 of the License, or
*     (at your option) any later version.
*
*     This program is distributed in the hope that it will be useful,
*     but WITHOUT ANY WARRANTY; without even the implied warranty of
*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*     GNU General Public License for more details.
*
*     You should have received a copy of the GNU General Public License
*     along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#version 330 core

layout (location = 0) in vec3 position;

flat out vec3 startPos;
out vec3 vertPos;

uniform mat4 projection;

void main()
{
    vec4 pos = projection * vec4(position, 1.0f);
    vertPos     = pos.xyz / pos.w;
    startPos    = vertPos;
    gl_Position = pos;
}
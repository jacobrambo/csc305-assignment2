// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

void raytrace_sphere()
{
    std::cout << "Simple ray tracer, one sphere with orthographic projection" << std::endl;

    const std::string filename("sphere_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the
    // unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // Intersect with the sphere
            // NOTE: this is a special case of a sphere centered in the origin and for orthographic rays aligned with the z axis
            Vector2d ray_on_xy(ray_origin(0), ray_origin(1));
            const double sphere_radius = 0.9;

            if (ray_on_xy.norm() < sphere_radius)
            {
                // The ray hit the sphere, compute the exact intersection point
                Vector3d ray_intersection(
                    ray_on_xy(0), ray_on_xy(1),
                    sqrt(sphere_radius * sphere_radius - ray_on_xy.squaredNorm()));

                // Compute normal at the intersection point
                Vector3d ray_normal = ray_intersection.normalized();

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_parallelogram()
{
    std::cout << "Simple ray tracer, one parallelogram with orthographic projection" << std::endl;

    const std::string filename("plane_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(0, 0.7, -10);
    const Vector3d pgram_v(1, 0.4, 0);

    const Vector3d p_normal = pgram_u.cross(pgram_v);

    const Vector3d a = pgram_origin;
    const Vector3d b = pgram_origin + pgram_u;
    const Vector3d c = pgram_origin + pgram_v;

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // We check if the ray intersects the parallelogram by solving the following equation:
            // ray_origin + t*ray_direction = pgram_origin + u*pgram_u + v*pgram_v
            // ray_origin = e
            // ray_direction = d

            const Vector3d e = ray_origin;
            const Vector3d d = camera_view_direction;

            // Calculate the values of u, v, and t using the equation in the textbook.
            Matrix3d u_matrix, v_matrix, t_matrix, A_matrix;
            u_matrix << a(0) - e(0), a(0) - c(0), d(0), a(1) - e(1), a(1) - c(1), d(1), a(2) - e(2), a(2) - c(2), d(2);
            v_matrix << a(0) - b(0), a(0) - e(0), d(0), a(1) - b(1), a(1) - e(1), d(1), a(2) - b(2), a(2) - e(2), d(2);
            t_matrix << a(0) - b(0), a(0) - c(0), a(0) - e(0), a(1) - b(1), a(1) - c(1), a(1) - e(1), a(2) - b(2), a(2) - c(2), a(2) - e(2);
            A_matrix << a(0) - b(0), a(0) - c(0), d(0), a(1) - b(1), a(1) - c(1), d(1), a(2) - b(2),a(2) - c(2), d(2);

            double det_u = u_matrix.determinant();
            double det_v = v_matrix.determinant();
            double det_t = t_matrix.determinant();
            double det_A = A_matrix.determinant();

            double u, v, t;
            u = det_u / det_A;
            v = det_v / det_A;
            t = det_t / det_A;

            // if the ray hits the parallelogram:
            if (t>0 && u>0 && v>0 && u<1 && v<1) {
                // Calculate the intersection point between the ray and the parallelogram
                Vector3d ray_intersection = ray_origin + t * ray_direction;
        
                // Compute the normal at the intersection point (which is the normal of the parallelogram)
                Vector3d ray_normal = p_normal.normalized();

                // Simple diffuse model
                C(i, j) = -(light_position - ray_intersection).normalized().dot(ray_normal);

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.0);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_perspective()
{
    std::cout << "Simple ray tracer, one parallelogram with perspective projection" << std::endl;

    const std::string filename("plane_perspective.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);
    const Vector3d center(0, 0, 1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(0, 0.7, -10);
    const Vector3d pgram_v(1, 0.4, 0);

    const Vector3d p_normal = pgram_u.cross(pgram_v);

    const Vector3d a = pgram_origin;
    const Vector3d b = pgram_origin + pgram_u;
    const Vector3d c = pgram_origin + pgram_v;


    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            
            // the ray_origin is the camera origin and the ray direction changes depending on the location of the current pixel.
            const Vector3d ray_origin = camera_origin;
            const Vector3d ray_direction = pixel_center - camera_origin;

            const Vector3d e = camera_origin;
            const Vector3d d = ray_direction;


            // again we solve for the values of u, v, and t using the formula in the textbook
            Matrix3d u_matrix, v_matrix, t_matrix, A_matrix;
            u_matrix << a(0) - e(0), a(0) - c(0), d(0), a(1) - e(1), a(1) - c(1), d(1), a(2) - e(2), a(2) - c(2), d(2);
            v_matrix << a(0) - b(0), a(0) - e(0), d(0), a(1) - b(1), a(1) - e(1), d(1), a(2) - b(2), a(2) - e(2), d(2);
            t_matrix << a(0) - b(0), a(0) - c(0), a(0) - e(0), a(1) - b(1), a(1) - c(1), a(1) - e(1), a(2) - b(2), a(2) - c(2), a(2) - e(2);
            A_matrix << a(0) - b(0), a(0) - c(0), d(0), a(1) - b(1), a(1) - c(1), d(1), a(2) - b(2), a(2) - c(2), d(2);

            double det_u = u_matrix.determinant();
            double det_v = v_matrix.determinant();
            double det_t = t_matrix.determinant();
            double det_A = A_matrix.determinant();

            double u, v, t;
            u = det_u / det_A;
            v = det_v / det_A;
            t = det_t / det_A;


            // we check if the ray hit the parallelogram
            if (t > 0 && u > 0 && v > 0 && u < 1 && v < 1)
            {
                // We compute the intersection point
                Vector3d ray_intersection = ray_origin + t * ray_direction;

                // We compute the normal at the intersection point
                Vector3d ray_normal = p_normal.normalized();

                // Simple diffuse model
                C(i, j) = -(light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.0);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_shading()
{
    std::cout << "Simple ray tracer, one sphere with different shading" << std::endl;

    const std::string filename("shading.png");
    MatrixXd R = MatrixXd::Zero(800, 800); // Store the red
    MatrixXd G = MatrixXd::Zero(800, 800); // Store the green
    MatrixXd B = MatrixXd::Zero(800, 800); // Store the blue
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / A.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / A.rows(), 0);

    //Sphere setup
    const Vector3d sphere_center(0, 0, 0);
    const double sphere_radius = 0.9;

    //material params
    const Vector3d diffuse_color(1, 0, 1);
    const double specular_exponent = 100;
    const Vector3d specular_color(0, 0, 1);

    // Single light source
    const Vector3d light_position(-1, 1, 1);
    double ambient = 0.1;

    for (unsigned i = 0; i < A.cols(); ++i)
    {
        for (unsigned j = 0; j < A.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = camera_origin;
            const Vector3d ray_direction = (pixel_center - camera_origin).normalized();

            // Intersect with the sphere
            const double sphere_radius = 0.9;

            // We solve for the perspective case using the quadratic formula to determine how many times
            // the ray hits the sphere

            Vector3d e = ray_origin;
            Vector3d d = ray_direction;

            double r = sphere_radius;

            double A1 = ray_direction.dot(ray_direction);
            double B1 = 2 * (ray_direction.dot(ray_origin - sphere_center));
            double C1 = ((ray_origin - sphere_center).dot(ray_origin - sphere_center)) - (sphere_radius * sphere_radius);

            double delta = sqrt(B1 * B1 - 4 * A1 * C1);

            // If the ray hits the sphere once or twice:
            if (delta >= 0)
            {
                // Compute the intersection point
                double t = (-B1 - delta) / (2 * A1);
                Vector3d ray_intersection = ray_origin + t * ray_direction;

                // Compute the normal at the intersection point
                Vector3d ray_normal = (ray_intersection - sphere_center).normalized();


                // set up the parameters like the ray tracing slides
                Vector3d n = ray_normal;
                Vector3d l = (light_position - ray_intersection).normalized();
                Vector3d v = (camera_origin - ray_intersection).normalized();
                Vector3d h = (v + l) / ((v + l).norm());

                // create a a RGB diffuse and RGB specular vector
                Vector3d diffuse = diffuse_color * l.dot(n);
                Vector3d specular = specular_color * pow(n.dot(h), specular_exponent);

                R(i, j) = ambient + diffuse[0] + specular[0];
                R(i, j) = std::max(R(i, j), 0.0);

                G(i, j) = ambient + diffuse[1] + specular[1];
                G(i, j) = std::max(G(i, j), 0.0);

                B(i, j) = ambient + diffuse[2] + specular[2];
                B(i, j) = std::max(B(i, j), 0.0);


                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(R, G, B, A, filename);
}

int main()
{

    raytrace_sphere();
    raytrace_parallelogram();
    raytrace_perspective();
    raytrace_shading();

    return 0;
}
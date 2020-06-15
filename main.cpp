#include <iostream>
#include <cmath>
#include "tgaimage.h"
#include "model.h"

//
// Draw mode
typedef enum DRAW_MODE_e {
    MODE_POINT = 0,
    MODE_LINE,
    MODE_TRIANGLE,

} DRAW_MODE;

DRAW_MODE gl_draw_mode = MODE_TRIANGLE;

//
void point(int x, int y, TGAImage &image, TGAColor color);

void line1(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color);
void line2(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color);
void line3(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color);

void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color);
void triangle2(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color);
void triangle3(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color);
void triangle4(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color);

Vec3f get_wieght_in_barycenty(Vec2i t0, Vec2i t1, Vec2i t2, Vec2i p);

Vec3f barycentric(Vec3f *pts, Vec3f P);
void triangle5(Vec3f *pts, float *zbuffer, TGAImage &render,
        TGAImage *texture, Vec2f *ptex,
        bool per_pixel_lighting, TGAColor vertextLightColor, Vec3f *pnorm, int lightCount, TGAColor *lightColor, Vec3f *lightDir);

void ModelViewMatrix(Vec3f const& eye, Vec3f const& center, Vec3f const& up, Matrix& ModelView);
void ProjectionMatrix(float d, Matrix& Projection);
void ViewPortMatrix(int x, int y, int w, int h, float d, Matrix& ViewPort);

//
TGAColor white(255, 255, 255);
TGAColor red(255, 0, 0);
TGAColor green(255, 255, 0);
TGAColor black(0, 0, 0);

//
Matrix  ModelView;
Matrix  Projection;
Matrix  ViewPort;

//
class IShader {
public:
    virtual ~IShader() { }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) = 0;
    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) = 0;

};

void Rasterize(Vec4f *pverts, IShader *shader, TGAImage& image, float *zbuffer);
/*
//
int main() {
//    std::cout << "begin to write..." << std::endl;
//
//    TGAImage image(100, 100, TGAImage::RGB);
//    TGAColor col(255, 0, 0);
//
//    for (int i = 0; i < 100; ++i) {
//        for (int j = 0; j < 100; ++j) {
//            image.set(i, j, col);
//        }
//    }
//
//    image.write_tga_file("./output.tga");
//
//    std::cout << "write done!" << std::endl;

    int width = 400;
    int height = 400;
    Vec3f light_dir[] = {
            Vec3f(0, 0, 1.0) // Most other angels, the result looks bad, i guess the details of the model is not enough.
    };
    TGAColor light_color[] = {
            TGAColor(255, 255, 255)
    };
    const int light_count = sizeof(light_dir) / sizeof(Vec3f);
    // before, i did not call normalize for each light, so the rendered image looked weird.
    for(int i=0;i<light_count;++i){
        light_dir[i].normalize();
    }

    bool per_pixel_lighting = true;

    TGAImage image(width, height, TGAImage::RGB);

//    line(13, 20, 80, 40, image, white);
//    line3(20, 13, 40, 80, image, red);
//    line3(80, 40, 13, 20, image, red);

    Model *model = new Model("/Users/wuwenxi/code/tgatest/cmake-build-debug/african_head.obj");
//    for (int i=0; i<model->nfaces(); i++) {
//        std::vector<int> face = model->face(i);
//        for (int j=0; j<3; j++) {
//            Vec3f v0 = model->vert(face[j]);
//            Vec3f v1 = model->vert(face[(j+1)%3]);
//            int x0 = (v0.x+1.)*width/2.;
//            int y0 = (v0.y+1.)*height/2.;
//            int x1 = (v1.x+1.)*width/2.;
//            int y1 = (v1.y+1.)*height/2.;
//            line3(x0, y0, x1, y1, image, red);
//        }
//    }

//    Vec2i t0[3] = {Vec2i(10, 70),   Vec2i(50, 160),  Vec2i(70, 80)};
//    Vec2i t1[3] = {Vec2i(180, 50),  Vec2i(150, 1),   Vec2i(70, 180)};
//    Vec2i t2[3] = {Vec2i(180, 150), Vec2i(120, 160), Vec2i(130, 180)};
//    triangle4(t0[0], t0[1], t0[2], image, red);
//    triangle4(t1[0], t1[1], t1[2], image, white);
//    triangle4(t2[0], t2[1], t2[2], image, green);

//    float min_value = std::numeric_limits<float>::min();

    float *zbuffer = new float[width * height];
    for (int i = 0; i < width * height; ++i) {
        // what the fuck, you know? if i use numeric_limits<float>::min(),
        // some of the triangles laid on the left and right side are gone.
        // i get the correct image only if i set the initial value to a "normal" value
        // like -9999.0f instead of THE std::numeric_limits<float>::min();, the reason is that
        // std::numeric_limits<float>::min() is very close to zero, so it causes error in comparison to zero.
        // or you can use -numeric_limits<float>::max())
        zbuffer[i] = -std::numeric_limits<float>::max();
    }

    ProjectionMatrix(-0.25, Projection);
    ViewPortMatrix(0, 0, width, height, 255.0f, ViewPort);
    ModelViewMatrix(Vec3f(0, 0, 1), Vec3f(0, 0, 0), Vec3f(0, 1, 0), ModelView);

//    std::cout << Projection << std::endl;
//    std::cout << ViewPort << std::endl;
//    std::cout << ModelView << std::endl;

    TGAImage texture;
    bool tex_read = texture.read_tga_file("/Users/wuwenxi/workspace/CG/PipelineRender/tgatest/cmake-build-debug/african_head_diffuse.tga");
    texture.flip_vertically();

    for (int i=0; i<model->nfaces(); i++) {
        std::vector<int> face = model->face(i);
        Vec2i screen_coords[3];
        Vec3f world_coords[3];
        Vec3f world_coords_cp[3];
        Vec2f vert_tex_coord[3];
        Vec3f vert_normal_coord[3];
        for (int j=0; j<3; j++) {
            Vec3f v = model->vert(face[j]);
            // if we use the perspective camera, we should calculate the projected x, y points upon the view
            // Assume that the camera is located at z-axis with 10 distance from the origin
            //float view_x = v.x / (1.0f - v.z / 5.0f);
            //float view_y = v.y / (1.0f - v.z / 5.0f);
            //screen_coords[j] =  Vec2i((view_x+1.)*width/2., (view_y+1.)*height/2.);
 //           screen_coords[j] = Vec2i((v.x+1.)*width/2., (v.y+1.)*height/2.);
            world_coords_cp[j] = v;
            Vec4f model_vertex = embed<4>(v);
            model_vertex = ViewPort * Projection * ModelView * model_vertex;
            // In the tutor, division by model_vertex[3] is in Projection Matrix,
            // But the result is same if we do it by the end of the chain of matrix operations.
            Vec3f view_vertex = proj<3>(model_vertex / model_vertex[3]);
            world_coords[j] = Vec3f(int(view_vertex.x), int(view_vertex.y), view_vertex.z);

     //       world_coords_cp[j] = v;
     //       v[0] = screen_coords[j][0];
     //       v[1] = screen_coords[j][1];
     //       world_coords[j]  = v;
            // i used to write it as such: model->uv(face(j),j);
            // in the result that most triangles on the faces are messed up.
            vert_tex_coord[j] = model->uv(i, j);
            vert_normal_coord[j] = model->normal(i, j);
        //    point(screen_coords[j][0], screen_coords[j][1], image, red);
        }

        //triangle4(screen_coords[0], screen_coords[1], screen_coords[2], image, TGAColor(rand()%255, rand()%255, rand()%255, 255));
        // orthology projection, so we can ignore z componet directly.
        //triangle5(world_coords, image, TGAColor(rand()%255, rand()%255, rand()%255, 255));
        // Per-vertex lighting
        //Vec3f n = cross(world_coords_cp[2]-world_coords_cp[0], world_coords_cp[1]-world_coords_cp[0]);
        //n.normalize();
        //float intensity = n*light_dir;
        //if (intensity>0) {
        //    triangle5(world_coords, zbuffer, image, (tex_read ? &texture : nullptr),
        //            vert_tex_coord, TGAColor(intensity*255, intensity*255, intensity*255, 255));
        //    triangle4(screen_coords[0], screen_coords[1], screen_coords[2],
        //              image, TGAColor(intensity*255, intensity*255, intensity*255, 255));
        //}

        //
        // Take both per-vertex and per-pixel lighting into consideratation.
        if (per_pixel_lighting) {

            triangle5(world_coords, zbuffer, image,
                      (tex_read ? &texture : nullptr), vert_tex_coord,
                      true, black, vert_normal_coord, light_count, light_color, light_dir);

        } else {

            TGAColor vertexLightColor = black;
            float vertexLightIntensity[3] = {1.0f, 1.0f, 1.0f};
            for (int i = 0; i < light_count; ++i) {
                Vec3f n = cross(world_coords_cp[2]-world_coords_cp[0], world_coords_cp[1]-world_coords_cp[0]);
                n.normalize();
                float intensity = n*light_dir[i];
                if (intensity > 0.0f) {
                    vertexLightIntensity[0] *= (intensity * light_color[i][0])/255.0f;
                    vertexLightIntensity[1] *= (intensity * light_color[i][1])/255.0f;
                    vertexLightIntensity[2] *= (intensity * light_color[i][2])/255.0f;
                }
            }
            vertexLightColor[0] = vertexLightIntensity[0] * 255.0f;
            vertexLightColor[1] = vertexLightIntensity[1] * 255.0f;
            vertexLightColor[2] = vertexLightIntensity[2] * 255.0f;

            triangle5(world_coords, zbuffer, image,
                      (tex_read ? &texture : nullptr), vert_tex_coord,
                      false, vertexLightColor, vert_normal_coord, light_count, light_color, light_dir);
        }
    }

    image.flip_vertically();
    image.write_tga_file("./output.tga");

    delete [] zbuffer;

    delete model;
    model = nullptr;

    return 0;
}
*/
//
void point(int x, int y, TGAImage &image, TGAColor color) {
    image.set(x, y, color);
}

void line1(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color) {
    for (int x=x0; x<=x1; x++) {
        float t = (x - x0) / (float) (x1 - x0);
        int y = y0 * (1. - t) + y1 * t;
        image.set(x, y, color);
    }
}

void line2(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color) {
    bool steep = false;
    if (std::abs(x0-x1)<std::abs(y0-y1)) { // if the line is steep, we transpose the image
        std::swap(x0, y0);
        std::swap(x1, y1);
        steep = true;
    }
    if (x0>x1) { // make it left−to−right
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    for (int x=x0; x<=x1; x++) {
        float t = (x-x0)/(float)(x1-x0);
        int y = y0*(1.-t) + y1*t;       // originally, it's y0 + (y1 - y0) * t, they are exactly the same thing.
        if (steep) {
            image.set(y, x, color); // if transposed, de−transpose
        } else {
            image.set(x, y, color);
        }
    }
}

void line3(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color) {
    bool steep = false;
    if (std::abs(x0-x1)<std::abs(y0-y1)) {
        std::swap(x0, y0);
        std::swap(x1, y1);
        steep = true;
    }
    if (x0>x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    int dx = x1-x0;
    int dy = y1-y0;
    float derror = std::abs(dy/float(dx));
    float error = 0;
    int y = y0;
    for (int x=x0; x<=x1; x++) {
        if (steep) {
            image.set(y, x, color);
        } else {
            image.set(x, y, color);
        }
        error += derror;
        if (error>.5) {
            y += (y1>y0?1:-1);
            error -= 1.;
        }
    }
}

void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color) {
    // sort the vertices, t0, t1, t2 lower−to−upper (bubblesort yay!)
    if (t0.y>t1.y) std::swap(t0, t1);
    if (t0.y>t2.y) std::swap(t0, t2);
    if (t1.y>t2.y) std::swap(t1, t2);
    line3(t0[0], t0[1], t1[0], t1[1], image, green);
    line3(t1[0], t1[1], t2[0], t2[1], image, green);
    line3(t2[0], t2[1], t0[0], t0[1], image, red);
}

void triangle2(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color) {
    // sort the vertices, t0, t1, t2 lower−to−upper (bubblesort yay!)
    if (t0.y>t1.y) std::swap(t0, t1);
    if (t0.y>t2.y) std::swap(t0, t2);
    if (t1.y>t2.y) std::swap(t1, t2);
    int total_height = t2.y-t0.y;
    for (int y=t0.y; y<=t1.y; y++) {
        int segment_height = t1.y-t0.y+1;
        float alpha = (float)(y-t0.y)/total_height;
        float beta  = (float)(y-t0.y)/segment_height; // be careful with divisions by zero
        Vec2i A = t0 + (t2-t0)*alpha;
        Vec2i B = t0 + (t1-t0)*beta;
        image.set(A.x, y, red);
        image.set(B.x, y, green);
    }
}

void triangle3(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color) {
    if (t0.y==t1.y && t0.y==t2.y) {
        return;
    }
    // sort the vertices, t0, t1, t2 lower−to−upper (bubblesort yay!)
    if (t0.y>t1.y) std::swap(t0, t1);
    if (t0.y>t2.y) std::swap(t0, t2);
    if (t1.y>t2.y) std::swap(t1, t2);
    int total_height = t2.y-t0.y;
    for (int y=t0.y; y<=t1.y; y++) {
        int segment_height = t1.y-t0.y+1;
        float alpha = (float)(y-t0.y)/total_height;
        float beta  = (float)(y-t0.y)/segment_height; // be careful with divisions by zero
        Vec2i A = t0 + (t2-t0)*alpha;
        Vec2i B = t0 + (t1-t0)*beta;
        if (A.x>B.x) std::swap(A, B);
        for (int j=A.x; j<=B.x; j++) {
            image.set(j, y, color); // attention, due to int casts t0.y+i != A.y
        }
    }
    for (int y=t1.y; y<=t2.y; y++) {
        int segment_height =  t2.y-t1.y+1;
        float alpha = (float)(y-t0.y)/total_height;
        float beta  = (float)(y-t1.y)/segment_height; // be careful with divisions by zero
        Vec2i A = t0 + (t2-t0)*alpha;
        Vec2i B = t1 + (t2-t1)*beta;
        if (A.x>B.x) std::swap(A, B);
        for (int j=A.x; j<=B.x; j++) {
            image.set(j, y, color); // attention, due to int casts t0.y+i != A.y
        }
    }
}

Vec3f get_wieght_in_barycenty(Vec2i t0, Vec2i t1, Vec2i t2, Vec2i p) {
    Vec3f u = cross(Vec3f(t2[0]-t0[0], t1[0]-t0[0], t0[0]-p[0]),
            Vec3f(t2[1]-t0[1], t1[1]-t0[1], t0[1]-p[1]));

    if (std::abs(u[2])<1) return Vec3f(-1,1,1);
    return Vec3f(1.f-(u.x+u.y)/u.z, u.x/u.z, u.y/u.z);
}

void triangle4(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color) {
    //
    float min_x = std::min(std::min(t0[0], t1[0]), t2[0]);
    float min_y = std::min(std::min(t0[1], t1[1]), t2[1]);
    float max_x = std::max(std::max(t0[0], t1[0]), t2[0]);
    float max_y = std::max(std::max(t0[1], t1[1]), t2[1]);

    const int image_width = image.get_width();
    const int image_height = image.get_height();

    if (min_x < 0.0f) {
        min_x = 0.0f;
    }
    if (min_y < 0.0f) {
        min_y = 0.0f;
    }
    if (max_x > image_width) {
        max_x = image_width;
    }
    if (max_y > image_height) {
        max_y = image_height;
    }

    //
    for (int i = min_x; i <= max_x; ++i) {
        for (int j = min_y; j <= max_y; ++j) {
            Vec3f bary = get_wieght_in_barycenty(t0, t1, t2, Vec2i(i, j));
            if (bary[0] < 0.0f || bary[1] < 0.0f || bary[2] < 0.0f) {
                continue;
            }

            image.set(i, j, color);
        }
    }
}

Vec3f barycentric(Vec3f *pts, Vec3f P) {
    Vec3f u = cross(Vec3f(pts[2][0]-pts[0][0], pts[1][0]-pts[0][0], pts[0][0]-P[0]),
            Vec3f(pts[2][1]-pts[0][1], pts[1][1]-pts[0][1], pts[0][1]-P[1]));
    /* `pts` and `P` has integer value as coordinates
       so `abs(u[2])` < 1 means `u[2]` is 0, that means
       triangle is degenerate, in this case return something with negative coordinates */
    if (std::abs(u[2])>1e-2)
        return Vec3f(1.f-(u.x+u.y)/u.z, u.y/u.z, u.x/u.z);
    return Vec3f(-1,1,1);
}

void triangle5(Vec3f *pts, float *zbuffer, TGAImage &render,
               TGAImage *texture, Vec2f *ptex,
               bool per_pixel_lighting, TGAColor vertextLightColor, Vec3f *pnorm, int lightCount, TGAColor *lightColor, Vec3f *lightDir) {

    Vec2f bboxmin( std::numeric_limits<float>::max(),  std::numeric_limits<float>::max());
    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    int width = render.get_width() - 1;
    int height = render.get_height() - 1;
    Vec2f clamp(width, height);
    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            bboxmin[j] = std::max(0.f,      std::min(bboxmin[j], pts[i][j]));
            bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], pts[i][j]));
        }
    }
    Vec3f P;
    for (P.x=bboxmin.x; P.x<=bboxmax.x; P.x++) {
        for (P.y=bboxmin.y; P.y<=bboxmax.y; P.y++) {
            Vec3f bc_screen  = barycentric(pts, P);
            if (bc_screen.x<0 || bc_screen.y<0 || bc_screen.z<0) continue;
            P.z = 0;
            for (int i=0; i<3; i++) P.z += pts[i][2]*bc_screen[i];
            if (zbuffer[int(P.x+P.y*width)]<P.z) {
                zbuffer[int(P.x+P.y*width)] = P.z;

                //
                TGAColor tex_color = white;
                if (texture != nullptr) {
                    float u = 0.0f;
                    float v = 0.0f;
                    for (int i = 0; i < 3; ++i) {
                        float fu = ptex[i][0];
                        float fv = ptex[i][1];
                        u += ptex[i][0] * bc_screen[i];
                        v += ptex[i][1] * bc_screen[i];
                    }
                    tex_color = texture->get((u*texture->get_width()), (v*texture->get_height()));
                }

                TGAColor light_color = vertextLightColor;
                if (per_pixel_lighting) {

                    Vec3f norm;norm[0] = 0.0f;norm[1] = 0.0f; norm[2] = 0.0f;
                    for (int j = 0; j < 3; ++j) {
                        norm = norm + pnorm[j] * bc_screen[j];
                    }
                    norm.normalize();

                    float pixelLightIntensity[3] = {1.0f, 1.0f, 1.0f};
                    for (int i = 0; i < lightCount; ++i) {
//                        lightDir[i][0] = -lightDir[i][0];lightDir[i][1] = -lightDir[i][1];lightDir[i][2] = -lightDir[i][2];
//                        float intensity = norm * lightDir[i];
//                        if ((intensity < 0.0f)) {
//                            norm[0] = -norm[0];norm[1] = -norm[1]; norm[2] = -norm[2];
//                            norm.normalize();
//                            intensity = norm * lightDir[i];
//                        }
                        Vec3f n = proj<3>((Projection*ModelView).invert_transpose()*embed<4>(norm)).normalize();
                        Vec3f l = proj<3>(Projection*ModelView  *embed<4>(lightDir[i]        )).normalize();
                        float intensity = n * l;
                        pixelLightIntensity[0] *= (intensity*lightColor[i][0]/255.0);
                        pixelLightIntensity[1] *= (intensity*lightColor[i][1]/255.0);
                        pixelLightIntensity[2] *= (intensity*lightColor[i][2]/255.0);
                    }

                    light_color[0] = pixelLightIntensity[0] * 255.0f;
                    light_color[1] = pixelLightIntensity[1] * 255.0f;
                    light_color[2] = pixelLightIntensity[2] * 255.0f;
                }

                TGAColor pixel_color;
                pixel_color[0] = ((float)tex_color[0]/255.0f) * light_color[0];
                pixel_color[1] = ((float)tex_color[1]/255.0f) * light_color[1];
                pixel_color[2] = ((float)tex_color[2]/255.0f) * light_color[2];
                render.set(P.x, P.y, pixel_color);
            }
        }
    }
}

//
void ModelViewMatrix(Vec3f const& eye, Vec3f const& center, Vec3f const& up, Matrix& ModelView) {
    Vec3f z = (eye-center).normalize();
    Vec3f x = cross(up,z).normalize();
    Vec3f y = cross(z,x).normalize();

    Matrix Minv = Matrix::identity();
    Matrix Tr   = Matrix::identity();
    for (int i=0; i<3; i++) {
        Minv[0][i] = x[i];
        Minv[1][i] = y[i];
        Minv[2][i] = z[i];
        Tr[i][3] = -center[i]; // ***NOTE: IS NOT Tr[i][3] = -eye[i]; ***
    }

    ModelView = Minv*Tr;
}

void ProjectionMatrix(float d, Matrix& Projection) {
    Projection = Matrix::identity();

    Projection[3][2] = d;
}

void ViewPortMatrix(int x, int y, int w, int h, float d, Matrix& ViewPort) {
    ViewPort = Matrix::identity();

    ViewPort[0][3] = x+w/2.f;
    ViewPort[1][3] = y+h/2.f;
    ViewPort[2][3] = d/2.0f;       // Use to clip z-axis in view frustum, if it is ignored, the result still looks no bad.

    ViewPort[0][0] = w/2.f;
    ViewPort[1][1] = h/2.f;
    ViewPort[2][2] = d/2.f;        // Use to clip z-axis in view frustum, if it is ignored, the result still looks no bad.
}

//
// implementations For some Shader samplers.
class TextureShader : public IShader {
public:
    TextureShader(Model *_model) : model(_model) {

    }

    virtual ~TextureShader() {

    }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) {
        varying_uv.set_col(ivertex, model->uv(iface, ivertex));
        varying_intensity[ivertex] = std::max(0.f, model->normal(iface, ivertex)*light_dir);
   //     norms[ivertex] = model->normal(iface, ivertex);
        Vec4f gl_Vertex = embed<4>(model->vert(iface, ivertex));
    //    float z = gl_Vertex[2];
        gl_Vertex = ViewPort*Projection*ModelView*gl_Vertex;
    //    gl_Vertex[2] = z;
        return gl_Vertex;
    }

    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) {
//        Vec3f norm;norm[0] = 0.0f;norm[1] = 0.0f; norm[2] = 0.0f;
//        for (int j = 0; j < 3; ++j) {
//            norm = norm + norms[j] * bar[j];
//        }
//        norm.normalize();
//
//        light_dir[0] = -light_dir[0];light_dir[1] = -light_dir[1];light_dir[2] = -light_dir[2];
//        float intensity = norm * light_dir;
//        if ((intensity < 0.0f)) {
//            norm[0] = -norm[0];norm[1] = -norm[1]; norm[2] = -norm[2];
//            norm.normalize();
//            intensity = norm * light_dir;
//        }

        float intensity = varying_intensity * bar;
        Vec2f uv = varying_uv*bar;
        pixelColor = model->diffuse(uv)*intensity;
        return false;
    }

public:
    inline void SetLightDirection(Vec3f const& lightDir) {
        light_dir = lightDir;
    }

private:
    Model *model;

    Vec3f varying_intensity;
    mat<2,3,float> varying_uv;

    mat<3,3,float> varying_tri;
    Vec3f          varying_ity;

//    Vec3f   norms[3];

    Vec3f light_dir;
};

class ToonShader : public IShader {
public:
    ToonShader(Model *_model) : model(_model) {

    }

    virtual ~ToonShader() {

    }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, ivertex));

//        light_dir[0] = -light_dir[0];light_dir[1] = -light_dir[1];light_dir[2] = -light_dir[2];
//        Vec3f norm = model->normal(iface, ivertex);
//        float intensity = norm * light_dir;
//        if ((intensity < 0.0f)) {
//            norm[0] = -norm[0];
//            norm[1] = -norm[1];
//            norm[2] = -norm[2];
//            norm.normalize();
//            intensity = norm * light_dir;
//        }

        varying_ity[ivertex] = std::max(0.f, model->normal(iface, ivertex)*light_dir);
        gl_Vertex = ViewPort*Projection*ModelView*gl_Vertex;
        return gl_Vertex;
    }

    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) {
        float intensity = varying_ity*bar;
        if (intensity>.85) intensity = 1;
        else if (intensity>.60) intensity = .80;
        else if (intensity>.45) intensity = .60;
        else if (intensity>.30) intensity = .45;
        else if (intensity>.15) intensity = .30;
        pixelColor = TGAColor(255, 155, 0)*intensity;
        return false;
    }

public:
    inline void SetLightDirection(Vec3f const& lightDir) {
        light_dir = lightDir;
    }

private:
    Model *model;

    mat<3,3,float> varying_tri;
    Vec3f          varying_ity;

    Vec3f light_dir;
};

class FlatShader : public IShader {
public:
    FlatShader(Model *_model) : model(_model) { }
    virtual ~FlatShader() { }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, ivertex));
        gl_Vertex = Projection*ModelView*gl_Vertex;
        varying_tri.set_col(ivertex, proj<3>(gl_Vertex/gl_Vertex[3]));
        gl_Vertex = ViewPort*gl_Vertex;
        return gl_Vertex;
    }

    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) {
        Vec3f n = cross(varying_tri.col(1)-varying_tri.col(0),varying_tri.col(2)-varying_tri.col(0)).normalize();
//        light_dir[0] = -light_dir[0];light_dir[1] = -light_dir[1];light_dir[2] = -light_dir[2];
//        intensity = n * light_dir;
//        if ((intensity < 0.0f)) {
//            n[0] = -n[0];
//            n[1] = -n[1];
//            n[2] = -n[2];
//            n.normalize();
//            intensity = n * light_dir;
//        }

        float intensity = std::max(0.0f, n*light_dir);
        pixelColor = TGAColor(255, 255, 255)*intensity;
        return false;
    }

public:
    inline void SetLightDirection(Vec3f const& lightDir) {
        light_dir = lightDir;
    }

private:
    Model *model;

    mat<3,3,float> varying_tri;

    float intensity;

    Vec3f light_dir;
};

class GouraudShader : public IShader {
public:
    GouraudShader(Model *_model) : model(_model) { }
    virtual ~GouraudShader() { }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, ivertex));
        gl_Vertex = ViewPort*Projection*ModelView*gl_Vertex;

//        Vec3f norm = model->normal(iface, ivertex);
//        light_dir[0] = -light_dir[0];light_dir[1] = -light_dir[1];light_dir[2] = -light_dir[2];
//        float intensity = norm * light_dir;
//        if ((intensity < 0.0f)) {
//            norm[0] = -norm[0];
//            norm[1] = -norm[1];
//            norm[2] = -norm[2];
//            norm.normalize();
//            intensity = norm * light_dir;
//        }
        Vec3f norm = model->normal(iface, ivertex);
        float intensity = std::max(0.0f, norm * light_dir);
        varying_intensity[ivertex] = intensity;

        return gl_Vertex;
    }

    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) {
        float intensity = varying_intensity*bar;
        pixelColor = TGAColor(255, 255, 255)*intensity;
        return false;
    }

public:
    inline void SetLightDirection(Vec3f const& lightDir) {
        light_dir = lightDir;
    }

private:
    Model *model;
    Vec3f light_dir;
    Vec3f varying_intensity;

};

class SmoothNormalShader : public IShader {
public:
    SmoothNormalShader(Model *_model) : model(_model) { }
    virtual ~SmoothNormalShader() { }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) {
        varying_uv.set_col(ivertex, model->uv(iface, ivertex));
        varying_normal[ivertex] = model->normal(iface, ivertex);
        Vec4f gl_Vertex = embed<4>(model->vert(iface, ivertex));
        gl_Vertex = ViewPort*Projection*ModelView*gl_Vertex;
        return gl_Vertex;
    }

    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) {
        Vec3f n;n[0]=0.0f;n[1]=0.0f;n[2]=0.0f;
        for(int i=0;i<3;++i){
            n=n+varying_normal[i]*bar[i];
        }
        n.normalize();
        float intensity = std::max(0.0f, n*light_dir);

        Vec2f uv = varying_uv*bar;
        pixelColor = model->diffuse(uv)*intensity;
        return false;
    }

public:
    inline void SetLightDirection(Vec3f const& lightDir) {
        light_dir = lightDir;
    }

private:
    Model *model;
    Vec3f light_dir;
    mat<2,3,float> varying_uv;
    Vec3f varying_normal[3];

};

class NormalMappingShader : public IShader { // First implementation of normal mapping in my life, 2020-06-14
public:
    mat<2,3,float> varying_uv;
    mat<4,4,float> uniform_M;
    mat<4,4,float> uniform_MIT;

public:
    NormalMappingShader(Model *_model) : model(_model) { }
    virtual ~NormalMappingShader() { }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) {
        varying_uv.set_col(ivertex, model->uv(iface, ivertex));
        Vec4f gl_Vertex = embed<4>(model->vert(iface, ivertex));
        gl_Vertex = ViewPort*Projection*ModelView*gl_Vertex;
        return gl_Vertex;
    }

    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) {
        Vec2f uv = varying_uv*bar;
        Vec3f norm = proj<3>(uniform_MIT*embed<4>(model->normal(uv))).normalize();
        Vec3f l = proj<3>(uniform_M  *embed<4>(light_dir        )).normalize();
        float intensity = norm * l;
        pixelColor = model->diffuse(uv)*intensity;
        return false;
    }

public:
    inline void SetLightDirection(Vec3f const& lightDir) {
        light_dir = lightDir;
    }

private:
    Model *model;
    Vec3f light_dir;

};

class PhongSpecularShader : public IShader {
public:
    mat<2,3,float> varying_uv;
    mat<4,4,float> uniform_M;
    mat<4,4,float> uniform_MIT;

public:
    PhongSpecularShader(Model *_model) : model(_model) { }
    virtual ~PhongSpecularShader() { }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) {
        varying_uv.set_col(ivertex, model->uv(iface, ivertex));
        Vec4f gl_Vertex = embed<4>(model->vert(iface, ivertex));
        gl_Vertex = ViewPort*Projection*ModelView*gl_Vertex;
        return gl_Vertex;
    }

    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) {
        Vec2f uv = varying_uv*bar;
        Vec3f n = proj<3>(uniform_MIT*embed<4>(model->normal(uv))).normalize();
        Vec3f l = proj<3>(uniform_M  *embed<4>(light_dir        )).normalize();
        Vec3f r = (n*(n*l*2.f) - l).normalize();
        float spec = pow(std::max(r.z, 0.0f), model->specular(uv));
        float diff = std::max(0.f, n*l);
        TGAColor diff_color = model->diffuse(uv);
        pixelColor = diff_color;
        for (int i=0; i<3; i++) {
            pixelColor[i] = std::min<float>(5 + diff_color[i]*(diff + .6*spec), 255);
        }

        return false;
    }

public:
    inline void SetLightDirection(Vec3f const& lightDir) {
        light_dir = lightDir;
    }

private:
    Model *model;
    Vec3f light_dir;

};

class BiTagentNormalShader : public IShader {
public:
    mat<2,3,float> varying_uv;
    mat<4,3,float> varying_tri;
    mat<3,3,float> varying_nrm;
    mat<3,3,float> ndc_tri;         // triangle in normalized device coordinates
    mat<4,4,float> uniform_M;
    mat<4,4,float> uniform_MIT;

public:
    BiTagentNormalShader(Model *_model) : model(_model) { }
    virtual ~BiTagentNormalShader() { }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) {
        varying_uv.set_col(ivertex, model->uv(iface, ivertex));
        varying_nrm.set_col(ivertex, proj<3>(uniform_MIT*embed<4>(model->normal(iface, ivertex), 0.f)));
        Vec4f gl_Vertex = uniform_M*embed<4>(model->vert(iface, ivertex));
        varying_tri.set_col(ivertex, gl_Vertex);
        ndc_tri.set_col(ivertex, proj<3>(gl_Vertex/gl_Vertex[3]));
        gl_Vertex = ViewPort*gl_Vertex;
        return gl_Vertex;
    }

    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) {
        Vec3f bn = (varying_nrm*bar).normalize();
        Vec2f uv = varying_uv*bar;

        mat<3,3,float> A;
        A[0] = ndc_tri.col(1) - ndc_tri.col(0);
        A[1] = ndc_tri.col(2) - ndc_tri.col(0);
        A[2] = bn;

        mat<3,3,float> AI = A.invert();

        Vec3f i = AI * Vec3f(varying_uv[0][1] - varying_uv[0][0], varying_uv[0][2] - varying_uv[0][0], 0);
        Vec3f j = AI * Vec3f(varying_uv[1][1] - varying_uv[1][0], varying_uv[1][2] - varying_uv[1][0], 0);

        mat<3,3,float> B;
        B.set_col(0, i.normalize());
        B.set_col(1, j.normalize());
        B.set_col(2, bn);

        Vec3f n = (B*model->normal(uv)).normalize();

        // in fact, the following line has the same result due to Projection matrix is nothing but
        // a identity matrix except for element [3][2] used to clip in view frustrum.
        // Vec3f l = proj<3>(ModelView  *embed<4>(light_dir        )).normalize()
        Vec3f l = proj<3>(uniform_M  *embed<4>(light_dir        )).normalize();
        float diff = std::max(0.f, n*l);
        pixelColor = model->diffuse(uv)*diff;

        return false;
    }

public:
    inline void SetLightDirection(Vec3f const& lightDir) {
        light_dir = lightDir;
    }

private:
    Model *model;
    Vec3f light_dir;

};

class DepthShader : public IShader {
public:
    DepthShader(Model *_model) : model(_model) { }
    virtual ~DepthShader() { }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, ivertex));
        gl_Vertex = ViewPort*Projection*ModelView*gl_Vertex;
        varying_tri.set_col(ivertex, proj<3>(gl_Vertex/gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) {
        Vec3f p = varying_tri*bar;
        pixelColor = TGAColor(255, 255, 255)*(p.z/255.0f);
        return false;
    }

private:
    Model *model;

    mat<3,3,float> varying_tri;

};

class ShadowShader : public IShader {
public:
    mat<4,4,float> uniform_M;
    mat<4,4,float> uniform_MIT;
    mat<4,4,float> uniform_Mshadow;
    mat<3,3,float> varying_nrm;
    mat<3,3,float> ndc_tri;
    mat<2,3,float> varying_uv;
    mat<3,3,float> varying_tri;

public:
    ShadowShader(Model *_model) : model(_model) { }
    virtual ~ShadowShader() { }

public:
    virtual Vec4f VertexShader(int iface, int ivertex) {
        varying_uv.set_col(ivertex, model->uv(iface, ivertex));
        varying_nrm.set_col(ivertex, proj<3>(uniform_MIT*embed<4>(model->normal(iface, ivertex), 0.f)));
        Vec4f gl_Vertex = ViewPort*Projection*ModelView*embed<4>(model->vert(iface, ivertex));
        ndc_tri.set_col(ivertex, proj<3>(gl_Vertex/gl_Vertex[3]));
        varying_tri.set_col(ivertex, proj<3>(gl_Vertex/gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool FragmentShader(Vec3f bar, TGAColor& pixelColor) {
        //
        Vec3f bn = (varying_nrm*bar).normalize();
        Vec2f uv = varying_uv*bar;
        mat<3,3,float> A;
        A[0] = ndc_tri.col(1) - ndc_tri.col(0);
        A[1] = ndc_tri.col(2) - ndc_tri.col(0);
        A[2] = bn;
        mat<3,3,float> AI = A.invert();
        Vec3f i = AI * Vec3f(varying_uv[0][1] - varying_uv[0][0], varying_uv[0][2] - varying_uv[0][0], 0);
        Vec3f j = AI * Vec3f(varying_uv[1][1] - varying_uv[1][0], varying_uv[1][2] - varying_uv[1][0], 0);
        mat<3,3,float> B;
        B.set_col(0, i.normalize());
        B.set_col(1, j.normalize());
        B.set_col(2, bn);

        Vec3f n = (B*model->normal(uv)).normalize();
        Vec3f l = proj<3>(uniform_M  *embed<4>(light_dir        )).normalize();

        Vec3f r = (n*(n*l*2.f) - l).normalize();

        Vec4f sb_p = uniform_Mshadow*embed<4>(varying_tri*bar);
        sb_p = sb_p/sb_p[3];
        int idx = int(sb_p[0]) + int(sb_p[1])*shadowWdith;
        float shadow = .3+.7*(shadowBuffer[idx]<sb_p[2]+43.34);

        float spec = pow(std::max(r.z, 0.0f), model->specular(uv));
        float diff = std::max(0.f, n*l);
        TGAColor diff_color = model->diffuse(uv)*diff;
        for (int i=0; i<3; i++) {
            pixelColor[i] = std::min<float>(20 + diff_color[i]*shadow*(1.2*diff + .6*spec), 255);
        }

        return false;
    }

public:
    inline void SetLightDirection(Vec3f const& lightDir) {
        light_dir = lightDir;
    }

    inline void SetShadowBuffer(float *_shadowBuffer, int width) {
        shadowBuffer = _shadowBuffer;
        shadowWdith = width;
    }

private:
    Model *model;
    Vec3f light_dir;
    float *shadowBuffer;
    int shadowWdith;

};

//
void Rasterize(Vec3f *pverts, IShader *shader, TGAImage& image, float *zbuffer) {
    Vec2f bboxmin( std::numeric_limits<float>::max(),  std::numeric_limits<float>::max());
    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    int width = image.get_width() - 1;
    int height = image.get_height() - 1;
    Vec2f clamp(width, height);
    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            bboxmin[j] = std::max(0.f,      std::min(bboxmin[j], pverts[i][j]));
            bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], pverts[i][j]));
        }
    }
    Vec3f P;
    TGAColor pixel_color(255,255,255);
    for (P.x=bboxmin.x; P.x<=bboxmax.x; P.x++) {
        for (P.y=bboxmin.y; P.y<=bboxmax.y; P.y++) {
            Vec3f bc_screen  = barycentric(pverts, P);
            if (bc_screen.x<0 || bc_screen.y<0 || bc_screen.z<0) {
                continue;
            }
            float frag_depth = 0.0f;
            for (int i=0; i<3; i++) {
                frag_depth += pverts[i][2] * bc_screen[i];
            }

            if (zbuffer[int(P.x+P.y*width)]<frag_depth) {
                bool discard_pixel = shader->FragmentShader(bc_screen, pixel_color);
                if (!discard_pixel) {
                    zbuffer[(int)(P.x+P.y*width)] = frag_depth;
                    image.set(P.x, P.y, pixel_color);
                }
            }
        }
    }

//    Vec2i P;
//    TGAColor color;
//    for (P.x=bboxmin.x; P.x<=bboxmax.x; P.x++) {
//        for (P.y=bboxmin.y; P.y<=bboxmax.y; P.y++) {
//            Vec3f c = BarycentricCoord(
//              proj<2>(pverts[0]/pverts[0][3]),
//              proj<2>(pverts[1]/pverts[1][3]),
//              proj<2>(pverts[2]/pverts[2][3]), proj<2>(P));
//            float z = pverts[0][2]*c.x + pverts[1][2]*c.y + pverts[2][2]*c.z;
//            float w = pverts[0][3]*c.x + pverts[1][3]*c.y + pverts[2][3]*c.z;
//            int frag_depth = std::max(0, std::min(255, int(z/w+.5)));
//            if (c.x<0 || c.y<0 || c.z<0) continue;
//            if (zbuffer[int(P.x+P.y*width)]<frag_depth) {
//                bool discard = shader->FragmentShader(c, color);
//                if (!discard) {
//                    zbuffer[P.x+P.y*image.get_width()] = frag_depth;
//                    image.set(P.x, P.y, color);
//                }
//            }
//        }
//    }

}

//
void Test_Shaders() {
    const int VIEW_WIDTH = 400;
    const int VIEW_HEIGHT = 400;

    Model *model = new Model("/Users/wuwenxi/workspace/CG/PipelineRender/tgatest/cmake-build-debug/african_head.obj");
    //Model *model = new Model("/Users/wuwenxi/code/git-project/tinyrenderer/obj/diablo3_pose/diablo3_pose.obj");

    Vec3f light_dir(1,1,1);
//    Vec3f light_dir(1,1,1);
    // The following 4 lines (including the comment lines) are very very important For understanding the camera view coordinat.
    // In the view frustrum the result are exactly the same as long as the view camera coordinates are same.(Right-handed herein)
    // Right-handed,so we do eye - center, actually the center is the origin of the camera system.
//    Vec3f       eye(0,0,0);
//    Vec3f    center(-0.2,-0.3,-1);
//    Vec3f       eye(0.2,0.3,1);
//    Vec3f       eye(0,0,1);
//    Vec3f   eye(0.2,0.3,1);
    Vec3f   eye(0.2,0.3,1);
    Vec3f    center(0,0,0);
    Vec3f        up(0,1,0);

    ModelViewMatrix(eye, center, up, ModelView);
    ProjectionMatrix(-0.05, Projection);
    ViewPortMatrix(0, 0, VIEW_WIDTH, VIEW_HEIGHT, 255.0f, ViewPort);

    // Did you notice that we did not set the light color thus the light is in white, because it equals to the final
    // pixel color multiply TGAColor(255, 255, 255)
    light_dir.normalize();

    //
    TGAImage image(VIEW_WIDTH, VIEW_HEIGHT, TGAImage::RGB);
    float *zbuffer = new float[VIEW_WIDTH*VIEW_HEIGHT];
    for (int i=0;i<VIEW_WIDTH*VIEW_HEIGHT;++i) {
        zbuffer[i] = -std::numeric_limits<float>::max();
    }

//    TextureShader shader(model);
//    ToonShader shader(model);
//    FlatShader shader(model);
//    GouraudShader shader(model);
//    SmoothNormalShader shader(model);
//    NormalMappingShader shader(model);
    PhongSpecularShader shader(model);
//    BiTagentNormalShader shader(model);
    shader.SetLightDirection(light_dir);
    shader.uniform_M   =  Projection*ModelView;
    shader.uniform_MIT = (Projection*ModelView).invert_transpose();
    for (int i=0; i<model->nfaces(); i++) {
        Vec3f screen_coords[3];
        for (int j=0; j<3; j++) {
            Vec4f temp = shader.VertexShader(i, j);
            screen_coords[j] = proj<3>(temp / temp[3]);
            screen_coords[j][0] = (int)screen_coords[j][0];
            screen_coords[j][1] = (int)screen_coords[j][1];
            if (gl_draw_mode == MODE_POINT) {
                point(screen_coords[j][0], screen_coords[j][1], image, TGAColor(255, 255, 255));

            }
        }
        if (gl_draw_mode == MODE_LINE) {
            line3(screen_coords[0][0], screen_coords[0][1], screen_coords[1][0], screen_coords[1][1], image, TGAColor(255, 255, 255));
            line3(screen_coords[1][0], screen_coords[1][1], screen_coords[2][0], screen_coords[2][1], image, TGAColor(255, 255, 255));
            line3(screen_coords[2][0], screen_coords[2][1], screen_coords[0][0], screen_coords[0][1], image, TGAColor(255, 255, 255));

        } else if (gl_draw_mode == MODE_TRIANGLE) {
            Rasterize(screen_coords, &shader, image, zbuffer);

        }
    }

    image.flip_vertically();
    image.write_tga_file("./output.tga");

    delete [] zbuffer;
    delete model;
}

// shadow rendering requires two-pass shader
void Test_ShadowShader() {
    // Set up
    const int VIEW_WIDTH = 400;
    const int VIEW_HEIGHT = 400;

    //Model *model = new Model("/Users/wuwenxi/workspace/CG/PipelineRender/tgatest/cmake-build-debug/african_head.obj");
    Model *model = new Model("/Users/wuwenxi/code/git-project/tinyrenderer/obj/diablo3_pose/diablo3_pose.obj");

    Vec3f light_dir(1,1,1);
    Vec3f       eye(0.5,0.2,3);
    Vec3f    center(0,0,0);
    Vec3f        up(0,1,0);

    float *shadowbuffer = new float[VIEW_WIDTH*VIEW_HEIGHT];
    float *zbuffer = new float[VIEW_WIDTH*VIEW_HEIGHT];
    for(int i=0;i<VIEW_WIDTH*VIEW_HEIGHT;++i){
        shadowbuffer[i]=-std::numeric_limits<float>::max();
        zbuffer[i]=-std::numeric_limits<float>::max();
    }

    light_dir.normalize();

    // Pass one: generate the shadow map
    {
        TGAImage shadowMap(VIEW_WIDTH, VIEW_HEIGHT, TGAImage::RGB);
        ModelViewMatrix(light_dir, center, up, ModelView);
        ViewPortMatrix(0, 0, VIEW_WIDTH, VIEW_HEIGHT, 255, ViewPort);
        ProjectionMatrix(0, Projection);

        DepthShader depthShader(model);
        Vec3f screen_coords[3];
        for (int i=0; i<model->nfaces(); i++) {
            for (int j=0; j<3; j++) {
                Vec4f temp = depthShader.VertexShader(i, j);
                screen_coords[j] = proj<3>(temp / temp[3]);
                screen_coords[j][0] = (int)screen_coords[j][0];
                screen_coords[j][1] = (int)screen_coords[j][1];
            }
            Rasterize(screen_coords, &depthShader, shadowMap, shadowbuffer);
        }
        shadowMap.flip_vertically();
        shadowMap.write_tga_file("./shadowmap.tga");
    }

    // Pass two: render the model with shadow map
    Matrix M = ViewPort*Projection*ModelView; // Save current one

    {
        TGAImage image(VIEW_WIDTH, VIEW_HEIGHT, TGAImage::RGB);
        // Setup New one
        ModelViewMatrix(eye, center, up, ModelView);
        ViewPortMatrix(0, 0, VIEW_WIDTH, VIEW_HEIGHT, 255, ViewPort);
        ProjectionMatrix(-1.f/(eye-center).norm(), Projection);

        // Call shader to render
        ShadowShader shadowShader(model);
        shadowShader.uniform_M = ModelView;
        shadowShader.uniform_MIT = (Projection*ModelView).invert_transpose();
        // M2 = M1*M => M = M1_Invert * M2
        // M2: M; M1: (Viewport*Projection*ModelView)
        shadowShader.uniform_Mshadow = M*(ViewPort*Projection*ModelView).invert();
        shadowShader.SetLightDirection(light_dir);
        shadowShader.SetShadowBuffer(shadowbuffer, VIEW_WIDTH);

        Vec3f screen_coords[3];
        for (int i=0; i<model->nfaces(); i++) {
            for (int j=0; j<3; j++) {
                Vec4f temp = shadowShader.VertexShader(i, j);
                screen_coords[j] = proj<3>(temp / temp[3]);
                screen_coords[j][0] = (int)screen_coords[j][0];
                screen_coords[j][1] = (int)screen_coords[j][1];
            }
            Rasterize(screen_coords, &shadowShader, image, zbuffer);
        }
        image.flip_vertically();
        image.write_tga_file("./Output.tga");
    }

    // release
    delete [] shadowbuffer;
    delete [] zbuffer;
    delete model;

}

//
// Test shaders
int main() {

//    Test_Shaders();
    Test_ShadowShader();

    return 0;
}

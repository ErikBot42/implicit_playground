struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) signed_uv: vec2<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    //camera.view_proj;
    var out: VertexOutput;
    out.color = model.color;
    //out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.clip_position = vec4<f32>(model.position, 1.0);
    out.uv = vec2<f32>(model.position.x, model.position.y) * vec2<f32>(0.5) + vec2<f32>(0.5);
    out.signed_uv = vec2<f32>(model.position.x, model.position.y);
    return out;
}

// srgb_color = ((rgb_color / 255 + 0.055) / 1.055) ^ 2.4
// TODO: try 16 bit float

// +-------+-------+
// |       |       |
// |       |       |
// +-------+-------+
// |       |       |
// |       |       |
// +-------+-------+ 

// cx = width*u;
// cy = height*v;
//
// cx/cy = (width/height) * (u/v)
// cx/cy * (v/u) = (width/height) 

// transformation matrix:
// if component 4 = 1 => do translation + rotation
// if component 4 = 0 => do rotation
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {


    // ray: point, unit dir
    let u = in.uv.x;
    let v = in.uv.y;

    // img
    let aspect_ratio = (in.clip_position.x/in.clip_position.y) / (u/(1.0-v)); // 16.0/9.0

    // cam
    let viewport_height = 2.0; 
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;
    
    let origin = vec3<f32>(0.0);
    let horizontal = vec3<f32>(viewport_width, 0.0, 0.0);
    let vertical  = vec3<f32>(0.0, viewport_height, 0.0);
    let lower_left_corner = origin - vec3<f32>(viewport_width, viewport_height, focal_length*2.0)/2.0;



    let ray_origin = origin;

    // TODO: inline aspects of this expr
    // TODO: omit norm if needed
    var ray_dir : vec3<f32>;
    ray_dir = origin - vec3<f32>(viewport_width, viewport_height, focal_length * 2.0) / 2.0 
        + vec3<f32>(u * viewport_width, v * viewport_height, 0.0) 
        - origin; 

    ray_dir = normalize(ray_dir);
    
    // sphere hit:
    let sphere_center = vec3<f32>(0.0, 0.0, -1.0);
    let sphere_radius = 0.5;

    let oc = ray_origin - sphere_center;
    let a = dot(ray_dir, ray_dir); // TODO: dot2, =1 iff normalized
    let half_b = dot(oc, ray_dir);
    let c = dot(oc, oc) - sphere_radius * sphere_radius;
    let disc = half_b * half_b - a * c;
    var t = 0.0;
    if (disc > 0.0) {
        t = (-half_b - sqrt(disc)) * a;
    } else {
        t = -1.0;
    }
    
    var col: vec3<f32>;
    let f = 10.0;
    if (t > 0.0) {
        let norm = normalize(ray_origin + t * ray_dir - vec3<f32>(0.0, 0.0, -1.0));
        let rdir = reflect(ray_dir, norm);
        col = 0.5*(vec3<f32>(norm)+1.0);
        let cdir = rdir;
        col *= sign(((atan2(cdir.x, cdir.z) * f + 10.0 * f) % 1.0 - 0.5) * ((acos(cdir.y) * f + 10.0 * f) % 1.0 - 0.5))*0.5+0.5;

        //let t = 0.5 * (dir.y + 1.0);
        //col += (1.0 - t) * vec3<f32>(1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
        //col *= 0.5;
    } else {
        let t = 0.5 * (ray_dir.y + 1.0);
        col = (1.0 - t) * vec3<f32>(1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
        let cdir = ray_dir;
        col *= sign(((atan2(cdir.x, cdir.z) * f + 10.0 * f) % 1.0 - 0.5) * ((acos(cdir.y) * f + 10.0 * f) % 1.0 - 0.5))*0.5+0.5;
    }

    var srgb_color: vec3<f32> = pow(col , vec3<f32>(2.4));
    //return vec4<f32>(
    //    in.clip_position.x/in.clip_position.y, 
    //    in.clip_position.y/in.clip_position.z, 
    //    in.clip_position.z/in.clip_position.x, 
    //    1.0
    //);
    return vec4<f32>(srgb_color, 1.0);
    //return vec4<f32>(1.0/aspect_ratio, 1.0, 0.0, 1.0);
    //return vec4<f32>(ray_dir, 1.0);
}

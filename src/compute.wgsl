// calc shadow map

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) 
var shadow_texture: texture_storage_2d<r32float, write>;

@group(0) @binding(1) 
var<uniform> camera: CameraUniform;

@compute
@workgroup_size(1) // does not need to execute in group
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i: u32 = global_id.x;
    let j: u32 = global_id.y;

    let u = f32(i)/100.0;
    let v = f32(j)/100.0;
    //let u = f32(i)/50.0 - 1.0;
    //let v = f32(j)/50.0 - 1.0;

    let uv = vec2<f32>(u, v);
    let fov = 1.0;

    let ray_dir = normalize(vec3<f32>(uv * fov, - 1.0));
    let ray_origin = vec3<f32>(0.0, 0.0, 0.0);
    
    let to = trace(ray_origin, ray_dir, 
        0.1 * length((camera.view_proj * vec4<f32>(vec3<f32>(0.0), 1.0)).xyz), 
        20.0, 200u, 0.001);
    let t: f32 = to.t;
    //let t = sin(u)*cos(v);
    //let t = abs(f32(i)/100.0);//abs(ray_dir.y);

    textureStore(shadow_texture, vec2<u32>(i, j), vec4<f32>(t));
}

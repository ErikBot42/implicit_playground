// calc shadow map


@group(0) @binding(0) 
var shadow_texture: texture_storage_2d<rgba16float, write>;
//var shadow_texture: texture_storage_2d<r32float, write>;
//var shadow_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1) 
var<uniform> camera: CameraUniform;

@compute
@workgroup_size(1) // does not need to execute in group
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let dim = vec2<f32>(textureDimensions(shadow_texture).xy);

    let ndim = 1.0/dim;

    let uv = vec2<f32>(vec2<f32>(global_id.xy) + vec2<f32>(0.5)) * ndim;

    let ray_dir = normalize(vec3<f32>(uv, - 1.0));
    let ray_origin = vec3<f32>(0.0, 0.0, 0.0);
    
    let to = trace_outer(
        ray_origin, 
        ray_dir, 
        0.1,// * length((camera.view_proj * vec4<f32>(vec3<f32>(0.0), 1.0)).xyz), 
        20.0, 
        200u, 
        //0.001
        max(ndim.x, ndim.y) * 2.0
    );
    let t: f32 = to.t + to.last_tol * 2.0;
    // - 2.0 * to.last_tol / dot(sdf_normal_e(ray_origin + ray_dir * to.t, to.last_tol), ray_dir);
    textureStore(shadow_texture, global_id.xy, vec4<f32>(t));
}

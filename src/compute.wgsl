// calc shadow map


@group(0) @binding(0) 
var shadow_texture: texture_storage_2d<r32float, write>;

@group(0) @binding(1) 
var<uniform> camera: CameraUniform;

@compute
@workgroup_size(1) // does not need to execute in group
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let uv = vec2<f32>(global_id.xy)/vec2<f32>(textureDimensions(shadow_texture).xy);

    let ray_dir = normalize(vec3<f32>(uv, - 1.0));
    let ray_origin = vec3<f32>(0.0, 0.0, 0.0);
    
    let to = trace_outer(
        ray_origin, 
        ray_dir, 
        0.1,// * length((camera.view_proj * vec4<f32>(vec3<f32>(0.0), 1.0)).xyz), 
        20.0, 
        200u, 
        0.001
    );
    let t: f32 = to.t;

    textureStore(shadow_texture, global_id.xy, vec4<f32>(t));
}

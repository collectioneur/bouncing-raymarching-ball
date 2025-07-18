import { useEffect, useMemo, useRef } from "react";
import { Dimensions } from "react-native";
import { Canvas, useDevice, useGPUContext } from "react-native-wgpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

const MAX_ITERATIONS = 50;
const MAX_DIST = 100.0;
const SURFACE_DIST = 0.01;

const mainVertex = tgpu["~unstable"].vertexFn({
  in: { vertexIndex: d.builtin.vertexIndex },
  out: { outPos: d.builtin.position, uv: d.vec2f },
})/* wgsl */ `{
  var pos = array<vec2f, 6>(vec2(-1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), 
    vec2(1.0, -1.0),
    vec2(1.0, 1.0));
  var uv = array<vec2f, 6>(
    vec2(0.0, 1.0), 
    vec2(0.0, 0.0),  
    vec2(1.0, 0.0),  
    vec2(0.0, 1.0),
    vec2(1.0, 0.0), 
    vec2(1.0, 1.0)   
  );
  return Out(vec4f(pos[in.vertexIndex], 0.0, 1.0), uv[in.vertexIndex]);
}`;

export default function BouncingBall() {
  const presentationFormat = (navigator as any)[
    "gpu"
  ].getPreferredCanvasFormat();
  const { device = null } = useDevice();
  const root = useMemo(
    () => (device ? tgpu.initFromDevice({ device }) : null),
    [device]
  );
  const { ref, context } = useGPUContext();
  const time = useMemo(() => root?.createUniform(d.f32) ?? null, [root]);
  const pipelineRef = useRef<GPURenderPipeline | null>(null);
  const { width, height } = Dimensions.get("window");
  const w = useMemo(() => root?.createUniform(d.f32) ?? null, [root]);
  const h = useMemo(() => root?.createUniform(d.f32) ?? null, [root]);

  useEffect(() => {
    if (!root || !device || !context) return;
    if (w === null || h === null || time === null) return;
    w.write(width);
    h.write(height);

    context.configure({
      device,
      format: presentationFormat,
      alphaMode: "premultiplied",
    });

    const palette = tgpu.fn([d.f32], d.vec3f)`(t) {
  let a = vec3f(0.71,0.08,0.69);
  let b = vec3f(0.50,0.05,0.31);
  let c = vec3f(0.69,0.08,0.89);
  let d = vec3f(0.81,0.63,0.58);

  return a + b * cos(6.28318 * (c * t + d));
}`;

    const sdSphere = tgpu.fn(
      [d.vec3f, d.f32],
      d.f32
    )((p, r) => {
      return std.length(p) - r;
    });

    const scene = tgpu.fn(
      [d.vec3f],
      d.f32
    )((p) => {
      let distance = sdSphere(p, 1.0);
      return distance;
    });

    const raymarch = tgpu.fn(
      [d.vec3f, d.vec3f],
      d.f32
    )((ro, rd) => {
      let dO = d.f32(0.0);
      for (let i = 0; i < MAX_ITERATIONS; i++) {
        let p = std.add(ro, std.mul(rd, dO));
        let dS = scene(p);
        dO = dO + dS;
        if (dS < SURFACE_DIST || dO > MAX_DIST) {
          break;
        }
      }
      return dO;
    });

    const getNormal = tgpu.fn(
      [d.vec3f],
      d.vec3f
    )((p) => {
      let e = d.vec2f(0.01, 0.0);
      let xyy = d.vec3f(e.x, e.y, e.y);
      let yxy = d.vec3f(e.y, e.x, e.y);
      let yyx = d.vec3f(e.y, e.y, e.x);

      let n = std.sub(
        scene(p),
        d.vec3f(
          scene(std.sub(p, xyy)),
          scene(std.sub(p, yxy)),
          scene(std.sub(p, yyx))
        )
      );
      return std.normalize(n);
    });

    const mainFragment = tgpu["~unstable"].fragmentFn({
      in: { uv: d.vec2f },
      out: d.vec4f,
    })(({ uv }) => {
      {
        let lightPos = d.vec3f(0.0, 0.0, -5.0);
        let new_uv = std.mul(std.sub(uv, 0.5), 2.0);
        new_uv.y *= h.$ / w.$;

        let ro = d.vec3f(
          std.cos(time.$),
          std.cos(time.$ * 4),
          -std.abs(std.sin(time.$ * 4)) * 5.0 - 1.0
        );
        let rd = std.normalize(d.vec3f(new_uv, 1.0));

        let dist = raymarch(ro, rd);
        let p = std.add(ro, std.mul(rd, dist));
        let color: d.v3f = d.vec3f(0.0, 0.0, 0.0);

        if (dist < MAX_DIST) {
          let normal = getNormal(p);
          let lightDir = std.normalize(std.sub(lightPos, p));
          let diff = std.max(0.0, std.dot(normal, lightDir));
          color = std.mul(diff, palette(dist / 10.0));
        }
        return d.vec4f(color, 1.0);
      }
    });

    const pipeline = root["~unstable"]
      .withVertex(mainVertex, {})
      .withFragment(mainFragment, { format: presentationFormat })
      .createPipeline();
    let startTime = performance.now();
    let frameId: number;

    const render = () => {
      const timestamp = (performance.now() - startTime) / 1000;
      if (timestamp > 500.0) {
        startTime = performance.now();
      }
      time.write(timestamp);

      const view = context.getCurrentTexture().createView();

      pipeline
        .withColorAttachment({
          view,
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        })
        .draw(6);

      context.present();
      frameId = requestAnimationFrame(render);
    };

    frameId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(frameId);
      // root.destroy();
    };
  }, [device, context]);

  return <Canvas ref={ref} style={{ flex: 1, backgroundColor: "black" }} />;
}

'use client'

/* eslint-disable react/no-unknown-property, react-hooks/immutability */
import {Canvas, useFrame, useThree} from '@react-three/fiber'
import type {ThreeEvent} from '@react-three/fiber'
import {useEffect, useMemo, useRef, useState, useSyncExternalStore} from 'react'
import * as THREE from 'three'

import * as aboutStyles from '@/components/about/about.styles'
import {collectTrace, currentChunkUrl} from '@/components/about/collectTrace'
import type {
  Bar,
  BarKind,
  PhaseKey,
  Trace,
} from '@/components/about/collectTrace'
import {TraceCaption, TraceLoading} from '@/components/about/LoadTraceShell'
import {useTraceText} from '@/components/about/traceText'
import type {TraceText} from '@/components/about/traceText'

// 모듈 평가 시점에 잡아야 스택 맨 위가 이 청크다
const SELF_URL = typeof window === 'undefined' ? undefined : currentChunkUrl()

const KIND_COLOR: Record<BarKind, string> = {
  document: '#f5f5fa',
  script: '#818cf8',
  css: '#2dd4bf',
  font: '#a78bfa',
  image: '#f472b6',
  other: '#8b8ba3',
}
const LONG_TASK_COLOR = '#fb7185'
const CURSOR_COLOR = '#2dd4bf'

const W = 10
const D = 5.6
const REPLAY_MS = 3600
const MAX_PARTICLES = 9000

interface Layout {
  laneGap: number
  // ms를 화면 x로, 재생 진행도(0..1)로, 그리고 진행도를 다시 ms로
  x: (t: number) => number
  u: (t: number) => number
  msAt: (u: number) => number
  total: number
  ticks: number[]
  gaps: {t: number; ms: number}[]
  bars: (Bar & {x0: number; x1: number; z: number; h: number})[]
  mainLaneZ: number
}

// 아무 요청도 없는 긴 공백은 짧은 틈으로 접는다. 대부분의 요청이 첫 200ms에 몰리고
// 몇 개만 한참 뒤에 오면, 선형 축은 가운데가 텅 빈다.
function layoutOf(trace: Trace): Layout {
  const spans = [
    ...trace.bars.map((b) => [b.start, b.end]),
    ...trace.marks.map((m) => [m.t, m.t]),
    ...trace.longTasks.map((t) => [t.start, t.end]),
  ].toSorted((a, b) => a[0] - b[0])
  const total = Math.max(...spans.map((sp) => sp[1]), 1)
  const idle = total * 0.12
  const folded = total * 0.035

  const knots: [number, number][] = [[0, 0]]
  const gaps: {t: number; ms: number}[] = []
  let lastT = 0
  let lastV = 0
  for (const [s, e] of spans) {
    if (e <= lastT) continue
    const from = Math.max(s, lastT)
    const gap = from - lastT
    if (gap > idle) {
      gaps.push({t: lastT + gap / 2, ms: gap})
      lastV += folded
    } else {
      lastV += gap
    }
    knots.push([from, lastV])
    lastV += e - from
    lastT = e
    knots.push([lastT, lastV])
  }
  const vEnd = lastV * 1.03

  const interp = (value: number, a: 0 | 1, b: 0 | 1) => {
    for (let i = 1; i < knots.length; i++) {
      const [p, q] = [knots[i - 1], knots[i]]
      if (value <= q[a]) {
        const span = q[a] - p[a]
        return span > 0 ? p[b] + ((value - p[a]) / span) * (q[b] - p[b]) : q[b]
      }
    }
    return knots.at(-1)![b]
  }
  const u = (t: number) => interp(t, 0, 1) / vEnd
  const x = (t: number) => -W / 2 + u(t) * W
  const msAt = (v: number) => interp(v * vEnd, 1, 0)

  const step = niceStep(lastV)
  const ticks: number[] = []
  for (let t = 0; t <= total; t += step) {
    if (!gaps.some((g) => Math.abs(t - g.t) < g.ms / 2)) ticks.push(t)
  }

  const laneGap = D / Math.max(trace.bars.length, 10)
  return {
    laneGap,
    x,
    u,
    msAt,
    total,
    ticks,
    gaps,
    mainLaneZ: D / 2 + laneGap * 3,
    bars: trace.bars.map((b, i) => ({
      ...b,
      x0: x(b.start),
      x1: Math.max(x(b.end), x(b.start) + 0.03),
      z: D / 2 - i * laneGap,
      h: 0.08 + Math.min(1, Math.log10(1 + b.bytes / 1024) * 0.36),
    })),
  }
}

const barVertex = /* glsl */ `
  attribute vec3 aColor;
  attribute vec2 aTime;
  attribute float aSelf;
  uniform float uCursor;
  uniform float uHover;
  uniform float uFocus;
  uniform float uFocusAmt;
  varying vec3 vColor;
  varying vec3 vNormalW;
  varying vec3 vViewW;
  varying float vTip;
  varying float vSelf;
  varying float vHover;
  varying float vY;
  varying float vDim;
  void main() {
    float span = max(aTime.y - aTime.x, 1e-4);
    float grow = clamp((uCursor - aTime.x) / span, 0.0, 1.0);
    vec3 p = position;
    p.x *= grow;
    p.y *= step(aTime.x, uCursor);
    vec4 world = modelMatrix * instanceMatrix * vec4(p, 1.0);
    vNormalW = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * normal);
    #ifdef MIRROR
      world.y = -world.y;
      vNormalW.y = -vNormalW.y;
    #endif
    vViewW = normalize(cameraPosition - world.xyz);
    vColor = aColor;
    vTip = (1.0 - step(1.0, grow)) * smoothstep(0.75, 1.0, position.x);
    vSelf = aSelf;
    vHover = 1.0 - step(0.5, abs(float(gl_InstanceID) - uHover));
    float focused = 1.0 - step(0.5, abs(float(gl_InstanceID) - uFocus));
    // 선택한 막대 말고는 가라앉힌다
    vDim = mix(1.0, mix(0.14, 1.5, focused), uFocusAmt);
    vY = position.y;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`

const barFragment = /* glsl */ `
  uniform float uTime;
  varying vec3 vColor;
  varying vec3 vNormalW;
  varying vec3 vViewW;
  varying float vTip;
  varying float vSelf;
  varying float vHover;
  varying float vY;
  varying float vDim;
  void main() {
    float rim = pow(1.0 - abs(dot(vNormalW, vViewW)), 2.2);
    float top = step(0.5, vNormalW.y);
    float body = 0.16 + 0.34 * top + 0.22 * vY;
    vec3 c = vColor * (body + rim * 1.25);
    c += vec3(1.0) * vTip * 0.9;
    c *= 1.0 + vSelf * (0.45 + 0.35 * sin(uTime * 3.2));
    c = mix(c, vColor * 1.6 + 0.25, vHover * 0.6);
    c *= vDim;
    #ifdef MIRROR
      c *= 0.28 * pow(1.0 - vY, 2.0);
    #endif
    gl_FragColor = vec4(c, 1.0);
  }
`

const dotVertex = /* glsl */ `
  attribute vec4 aLane;
  attribute vec2 aTime;
  attribute vec3 aColor;
  attribute float aSeed;
  attribute float aBar;
  uniform float uTime;
  uniform float uCursor;
  uniform float uPixelRatio;
  uniform float uFocus;
  uniform float uFocusAmt;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float span = max(aTime.y - aTime.x, 1e-4);
    float grow = clamp((uCursor - aTime.x) / span, 0.0, 1.0);
    float lit = step(aTime.x, uCursor);
    float speed = 0.18 + fract(aSeed * 7.13) * 0.35;
    float prog = fract(uTime * speed + aSeed);
    float x = mix(aLane.x, mix(aLane.x, aLane.y, grow), prog);
    float y = aLane.w + 0.05 + 0.06 * sin(uTime * 2.3 + aSeed * 40.0);
    float z = aLane.z + (fract(aSeed * 91.7) - 0.5) * 0.05;
    vec4 mv = modelViewMatrix * vec4(x, y, z, 1.0);
    gl_PointSize = (3.4 + fract(aSeed * 13.1) * 3.6) * uPixelRatio * (9.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
    vColor = aColor;
    float focused = 1.0 - step(0.5, abs(aBar - uFocus));
    vAlpha = lit * smoothstep(0.0, 0.08, prog) * (1.0 - smoothstep(0.85, 1.0, prog))
      * mix(1.0, mix(0.08, 1.4, focused), uFocusAmt);
  }
`

const dotFragment = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.0, d) * vAlpha;
    gl_FragColor = vec4(vColor * a * 1.6, a);
  }
`

const floorVertex = /* glsl */ `
  varying vec3 vWorld;
  void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`

// 눈금은 압축된 축을 따라야 해서 메시로 그리고, 바닥은 커서가 지나가는 빛만 맡는다
const floorFragment = /* glsl */ `
  uniform float uCursorX;
  uniform float uCursorOn;
  varying vec3 vWorld;
  void main() {
    float fade = 1.0 - smoothstep(2.0, 7.5, length(vWorld.xz * vec2(0.8, 1.2)));
    float sweep = exp(-abs(vWorld.x - uCursorX) * 6.0) * uCursorOn;
    gl_FragColor = vec4(vec3(0.18, 0.83, 0.75) * sweep * 0.35 * fade, 1.0);
  }
`

const cursorVertex = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

const cursorFragment = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity;
  varying vec2 vUv;
  void main() {
    float core = exp(-abs(vUv.x - 0.5) * 18.0);
    float fall = pow(1.0 - vUv.y, 1.6);
    float edge = smoothstep(0.0, 0.12, vUv.y) * smoothstep(0.0, 0.08, 1.0 - abs(vUv.x - 0.5) * 2.0);
    gl_FragColor = vec4(uColor * core * fall * edge * uOpacity * 1.4, 1.0);
  }
`

function niceStep(ms: number) {
  const raw = ms / 8
  const pow = 10 ** Math.floor(Math.log10(raw))
  return [1, 2, 5, 10].map((m) => m * pow).find((s) => s >= raw) ?? raw
}

// R3F의 <shaderMaterial uniforms>는 유니폼을 복사해 두므로(9.7 기준) 나중에 바꾼 값이
// 셰이더에 닿지 않는다. 머티리얼을 직접 만들어 유니폼 객체의 참조를 유지한다.
function glow(
  vertexShader: string,
  fragmentShader: string,
  uniforms: Record<string, THREE.IUniform>,
  extra: THREE.ShaderMaterialParameters = {},
) {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    ...extra,
  })
}

function Bars({
  layout,
  material,
  mirror,
  onHover,
  onSelect,
}: {
  layout: Layout
  material: THREE.ShaderMaterial
  mirror: THREE.ShaderMaterial
  onHover: (i: number, e?: ThreeEvent<PointerEvent>) => void
  onSelect: (i: number) => void
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const mirrorRef = useRef<THREE.InstancedMesh>(null)
  const geometry = useMemo(() => {
    const g = new THREE.BoxGeometry(1, 1, 1)
    g.translate(0.5, 0.5, 0)
    const n = layout.bars.length
    const color = new Float32Array(n * 3)
    const time = new Float32Array(n * 2)
    const self = new Float32Array(n)
    const c = new THREE.Color()
    layout.bars.forEach((b, i) => {
      c.set(KIND_COLOR[b.kind]).toArray(color, i * 3)
      time[i * 2] = layout.u(b.start)
      time[i * 2 + 1] = layout.u(b.end)
      self[i] = b.self ? 1 : 0
    })
    g.setAttribute('aColor', new THREE.InstancedBufferAttribute(color, 3))
    g.setAttribute('aTime', new THREE.InstancedBufferAttribute(time, 2))
    g.setAttribute('aSelf', new THREE.InstancedBufferAttribute(self, 1))
    return g
  }, [layout])

  useEffect(() => {
    const m = new THREE.Matrix4()
    for (const mesh of [meshRef.current, mirrorRef.current]) {
      if (!mesh) continue
      layout.bars.forEach((b, i) => {
        m.makeScale(b.x1 - b.x0, b.h, layout.laneGap * 0.62)
        m.setPosition(b.x0, 0, b.z)
        mesh.setMatrixAt(i, m)
      })
      mesh.instanceMatrix.needsUpdate = true
      mesh.computeBoundingSphere()
    }
  }, [layout])

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[geometry, material, layout.bars.length]}
        onPointerMove={(e) => {
          e.stopPropagation()
          onHover(e.instanceId ?? -1, e)
        }}
        onPointerOut={() => onHover(-1)}
        onClick={(e) => {
          e.stopPropagation()
          onSelect(e.instanceId ?? -1)
        }}
      />
      {/* 바닥 반사: 같은 기둥을 셰이더에서 뒤집어 흐리게 그린다 */}
      <instancedMesh
        ref={mirrorRef}
        args={[geometry, mirror, layout.bars.length]}
        frustumCulled={false}
        raycast={() => null}
      />
    </>
  )
}

function Bytes({
  layout,
  material,
}: {
  layout: Layout
  material: THREE.ShaderMaterial
}) {
  const geometry = useMemo(() => {
    const totalKb = layout.bars.reduce((n, b) => n + b.bytes / 2048, 0)
    const scale = Math.min(1, MAX_PARTICLES / Math.max(totalKb, 1))
    const counts = layout.bars.map((b) =>
      Math.max(2, Math.round((b.bytes / 2048) * scale)),
    )
    const n = counts.reduce((a, b) => a + b, 0)
    const lane = new Float32Array(n * 4)
    const time = new Float32Array(n * 2)
    const color = new Float32Array(n * 3)
    const seed = new Float32Array(n)
    const bar = new Float32Array(n)
    const c = new THREE.Color()
    let k = 0
    layout.bars.forEach((b, i) => {
      c.set(KIND_COLOR[b.kind])
      for (let j = 0; j < counts[i]; j++, k++) {
        lane.set([b.x0, b.x1, b.z, b.h], k * 4)
        time.set([layout.u(b.start), layout.u(b.end)], k * 2)
        c.toArray(color, k * 3)
        // 렌더마다 같은 값이 나오도록 인덱스에서 의사 난수를 만든다
        const r = Math.sin(k * 12.9898) * 43758.5453
        seed[k] = r - Math.floor(r)
        bar[k] = i
      }
    })
    const g = new THREE.BufferGeometry()
    g.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(n * 3), 3),
    )
    g.setAttribute('aLane', new THREE.BufferAttribute(lane, 4))
    g.setAttribute('aTime', new THREE.BufferAttribute(time, 2))
    g.setAttribute('aColor', new THREE.BufferAttribute(color, 3))
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    g.setAttribute('aBar', new THREE.BufferAttribute(bar, 1))
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 20)
    return g
  }, [layout])

  return (
    <points geometry={geometry} material={material} frustumCulled={false} />
  )
}

interface LabelSpec {
  key: string
  pos: THREE.Vector3
  text: string
  tone: 'mark' | 'tick' | 'self' | 'lane'
  at: number
}

// 드래그로 더한 회전과 확대, 이동. 재생 연출이 정하는 기본 구도 위에 얹는다.
interface Orbit {
  yaw: number
  pitch: number
  vYaw: number
  vPitch: number
  // 카메라 거리 배율. 1이 기본 구도
  zoom: number
  // 바라보는 점을 옮긴 양(월드 좌표)
  pan: THREE.Vector3
  // 아직 pan에 반영하지 않은 Shift 드래그 픽셀
  panDx: number
  panDy: number
  dragging: boolean
  panning: boolean
  // 더블클릭 뒤 기본 구도로 미끄러져 돌아가는 중
  homing: boolean
  touched: boolean
  invalidate: () => void
}

const FOV = 30
const ZOOM_MIN = 0.35
const ZOOM_MAX = 2.5

function resetOrbit(o: Orbit) {
  Object.assign(o, {
    yaw: 0,
    pitch: 0,
    vYaw: 0,
    vPitch: 0,
    zoom: 1,
    panDx: 0,
    panDy: 0,
    homing: false,
  })
  o.pan.set(0, 0, 0)
}

function Scene({
  trace,
  layout,
  replayKey,
  reduced,
  labels,
  labelEls,
  hudEl,
  onHover,
  hover,
  orbit,
  focus,
  onSelect,
  onFirstFrame,
}: {
  trace: Trace
  layout: Layout
  replayKey: number
  reduced: boolean
  labels: LabelSpec[]
  labelEls: React.RefObject<(HTMLDivElement | null)[]>
  hudEl: React.RefObject<HTMLDivElement | null>
  onHover: (i: number, e?: ThreeEvent<PointerEvent>) => void
  hover: number
  orbit: React.RefObject<Orbit>
  focus: number
  onSelect: (i: number) => void
  onFirstFrame: () => void
}) {
  const firstFrame = useRef(true)
  const {camera, size, gl, pointer, invalidate} = useThree()
  const sph = useMemo(() => new THREE.Spherical(), [])
  // 클릭한 막대로 초점을 옮긴다. 풀 때도 그 자리에서 미끄러져 돌아오도록 마지막 위치를 남긴다.
  const focusAmt = useRef(0)
  const focusPos = useMemo(() => new THREE.Vector3(), [])
  const focusDist = useRef(3)
  const aim = useMemo(() => new THREE.Vector3(), [])
  const cursorRef = useRef<THREE.Mesh>(null)
  const startRef = useRef<number | null>(null)
  const keyRef = useRef(replayKey)
  // 눈금 라벨은 x 순서대로 배치해야 겹침을 판단할 수 있다
  const tickOrder = useMemo(
    () =>
      labels
        .map((l, i) => i)
        .toSorted((a, b) => labels[a].pos.x - labels[b].pos.x),
    [labels],
  )
  const v = useMemo(() => new THREE.Vector3(), [])
  const look = useMemo(() => new THREE.Vector3(0, -0.15, -0.1), [])
  const dirStart = useMemo(
    () => new THREE.Vector3(-0.62, 0.18, 1).normalize(),
    [],
  )
  const dirEnd = useMemo(
    () => new THREE.Vector3(-0.16, 0.66, 1).normalize(),
    [],
  )
  const dir = useMemo(() => new THREE.Vector3(), [])
  const camRight = useMemo(() => new THREE.Vector3(), [])
  const camUp = useMemo(() => new THREE.Vector3(), [])

  const shared = useMemo(
    () => ({
      uCursor: {value: reduced ? 1 : 0},
      uTime: {value: 0},
      uHover: {value: -1},
      uFocus: {value: -1},
      uFocusAmt: {value: 0},
      uPixelRatio: {value: gl.getPixelRatio()},
    }),
    [gl, reduced],
  )
  const floor = useMemo(
    () => ({
      uCursorX: {value: -W / 2},
      uCursorOn: {value: 0},
    }),
    [],
  )
  const cursorUniforms = useMemo(
    () => ({
      uColor: {value: new THREE.Color(CURSOR_COLOR)},
      uOpacity: {value: 0},
    }),
    [],
  )
  const materials = useMemo(
    () => ({
      bars: glow(barVertex, barFragment, shared),
      mirror: glow(barVertex, barFragment, shared, {
        defines: {MIRROR: ''},
        side: THREE.DoubleSide,
      }),
      bytes: glow(dotVertex, dotFragment, shared),
      floor: glow(floorVertex, floorFragment, floor),
      cursor: glow(cursorVertex, cursorFragment, cursorUniforms, {
        side: THREE.DoubleSide,
      }),
      tick: new THREE.MeshBasicMaterial({
        color: '#2a2a3d',
        blending: THREE.AdditiveBlending,
        transparent: true,
        depthWrite: false,
      }),
    }),
    [shared, floor, cursorUniforms],
  )
  const markMaterials = useMemo(
    () =>
      trace.marks.map(
        () =>
          new THREE.MeshBasicMaterial({
            color: '#c9c9d6',
            transparent: true,
            opacity: 0,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
          }),
      ),
    [trace],
  )

  useEffect(() => {
    shared.uHover.value = hover
  }, [hover, shared])

  useEffect(() => {
    orbit.current.invalidate = invalidate
  }, [orbit, invalidate])

  useFrame((state, delta) => {
    if (firstFrame.current) {
      firstFrame.current = false
      onFirstFrame()
    }
    const now = state.clock.elapsedTime * 1000
    if (keyRef.current !== replayKey) {
      keyRef.current = replayKey
      startRef.current = null
      resetOrbit(orbit.current)
    }
    startRef.current ??= now
    const p = reduced ? 1 : Math.min(1, (now - startRef.current) / REPLAY_MS)
    const cursor = p
    shared.uCursor.value = cursor
    shared.uTime.value = reduced ? 1.7 : state.clock.elapsedTime

    const cx = -W / 2 + cursor * W
    floor.uCursorX.value = cx
    floor.uCursorOn.value =
      p < 1 ? 1 : Math.max(0, floor.uCursorOn.value - 0.02)
    cursorUniforms.uOpacity.value = floor.uCursorOn.value
    if (cursorRef.current) {
      cursorRef.current.position.x = cx
    }

    trace.marks.forEach((m, i) => {
      const since = cursor - layout.u(m.t)
      markMaterials[i].opacity =
        since < 0 ? 0 : 0.12 + 1.4 * Math.exp(-since / 0.05)
    })

    // 카메라: 낮고 가까이서 시작해 물러나며 전체를 드러낸다
    const aspect = size.width / size.height
    const fit = Math.max(1, 1.55 / aspect)
    const ease = 1 - (1 - p) ** 3
    dir.lerpVectors(dirStart, dirEnd, ease).normalize()

    const o = orbit.current
    if (!o.dragging && !reduced) {
      o.yaw += o.vYaw
      o.pitch += o.vPitch
      o.vYaw *= 0.88
      o.vPitch *= 0.88
    }
    if (o.homing) {
      const k = reduced ? 1 : 1 - Math.exp(-delta * 6)
      o.yaw *= 1 - k
      o.pitch *= 1 - k
      o.zoom += (1 - o.zoom) * k
      o.pan.multiplyScalar(1 - k)
      const left =
        Math.abs(o.yaw) +
        Math.abs(o.pitch) +
        Math.abs(o.zoom - 1) +
        o.pan.length()
      if (left < 0.002) resetOrbit(o)
      else invalidate()
    }
    o.pitch = THREE.MathUtils.clamp(o.pitch, -0.55, 0.7)
    sph.setFromVector3(dir)
    sph.theta -= o.yaw
    // 바닥 아래로는 내려가지 않는다
    sph.phi = THREE.MathUtils.clamp(sph.phi + o.pitch, 0.2, 1.46)
    dir.setFromSpherical(sph)

    if (focus >= 0) {
      const b = layout.bars[focus]
      focusPos.set((b.x0 + b.x1) / 2, b.h / 2, b.z)
      focusDist.current = Math.max(2.6, (b.x1 - b.x0) * 0.9 + 1.8)
      shared.uFocus.value = focus
    }
    const ramp = reduced ? 1 : 1 - Math.exp(-delta * 4)
    focusAmt.current += ((focus >= 0 ? 1 : 0) - focusAmt.current) * ramp
    const fa = focusAmt.current
    shared.uFocusAmt.value = fa
    if (fa > 0.001 && fa < 0.999) invalidate()

    const dist = THREE.MathUtils.lerp(
      THREE.MathUtils.lerp(6, 11.5, ease) * fit * o.zoom,
      focusDist.current,
      fa,
    )
    // Shift 드래그 픽셀을 지난 프레임 카메라의 가로세로 축으로 옮긴다
    if (o.panDx || o.panDy) {
      const perPx =
        (2 * dist * Math.tan(THREE.MathUtils.degToRad(FOV / 2))) / size.height
      camRight.setFromMatrixColumn(camera.matrixWorld, 0)
      camUp.setFromMatrixColumn(camera.matrixWorld, 1)
      o.pan
        .addScaledVector(camRight, -o.panDx * perPx)
        .addScaledVector(camUp, o.panDy * perPx)
      if (o.pan.length() > W) o.pan.setLength(W)
      o.panDx = 0
      o.panDy = 0
    }
    // 초점 모드에서는 확대와 이동을 접고 막대 구도로 간다
    aim.copy(look).add(o.pan).lerp(focusPos, fa)
    camera.position.copy(dir).multiplyScalar(dist).add(aim)
    if (!reduced && !o.touched) {
      camera.position.x += pointer.x * 0.8 * (1 - fa)
      camera.position.y += pointer.y * 0.4 * (1 - fa)
    }
    camera.lookAt(aim)

    // 3D 좌표를 화면에 투영해 HTML 라벨을 옮긴다
    let lastTickRight = -Infinity
    for (const i of tickOrder) {
      const l = labels[i]
      const el = labelEls.current[i]
      if (!el) continue
      v.copy(l.pos).project(camera)
      const x = (v.x * 0.5 + 0.5) * size.width
      const y = (-v.y * 0.5 + 0.5) * size.height
      // 확대해서 볼 때는 표지와 눈금을 숨긴다
      let on = cursor >= l.at && v.z < 1 && (fa < 0.3 || l.tone === 'self')
      if (l.tone === 'tick' && on) {
        const half = l.text.length * 3.6
        // 옆 라벨과 겹치거나 패널 가장자리를 넘으면 숨긴다
        if (
          x - half < lastTickRight + 8 ||
          x - half < 4 ||
          x + half > size.width - 4
        )
          on = false
        else lastTickRight = x + half
      }
      el.style.transform = `translate(${x}px, ${y}px)`
      el.style.opacity = on ? '1' : '0'
    }
    if (hudEl.current) {
      hudEl.current.textContent = `${Math.round(layout.msAt(cursor)).toLocaleString()} ms`
    }
  })

  return (
    <>
      <mesh
        rotation-x={-Math.PI / 2}
        position={[0, -0.001, 0]}
        material={materials.floor}
      >
        <planeGeometry args={[W + 3, D + 4]} />
      </mesh>
      <Bars
        layout={layout}
        material={materials.bars}
        mirror={materials.mirror}
        onHover={onHover}
        onSelect={onSelect}
      />
      <Bytes layout={layout} material={materials.bytes} />
      {trace.longTasks.map((t) => (
        <mesh
          key={t.start}
          position={[
            (layout.x(t.start) + layout.x(t.end)) / 2,
            0.012,
            layout.mainLaneZ,
          ]}
        >
          <boxGeometry
            args={[
              Math.max(layout.x(t.end) - layout.x(t.start), 0.02),
              0.024,
              layout.laneGap * 1.4,
            ]}
          />
          <meshBasicMaterial
            color={LONG_TASK_COLOR}
            transparent
            opacity={0.85}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      ))}
      {layout.ticks.map((t) => (
        <mesh
          key={t}
          position={[layout.x(t), 0.001, 0.25]}
          material={materials.tick}
        >
          <boxGeometry args={[0.008, 0.002, D + 0.9]} />
        </mesh>
      ))}
      {trace.marks.map((m, i) => (
        <group key={m.label} position={[layout.x(m.t), 0, 0]}>
          {/* 바닥을 가로지르는 빛줄과 뒤쪽 세로선 */}
          <mesh position={[0, 0.003, 0]} material={markMaterials[i]}>
            <boxGeometry args={[0.018, 0.006, D + 0.8]} />
          </mesh>
          <mesh position={[0, 0.55, -D / 2 - 0.4]} material={markMaterials[i]}>
            <boxGeometry args={[0.012, 1.1, 0.012]} />
          </mesh>
        </group>
      ))}
      <mesh
        ref={cursorRef}
        position={[-W / 2, 1.2, 0]}
        rotation-y={Math.PI / 2}
        material={materials.cursor}
      >
        <planeGeometry args={[D + 2, 2.4]} />
      </mesh>
    </>
  )
}

function transferText(bar: Bar, t: TraceText) {
  if (bar.cached) return t.cached
  return bar.transfer ? t.transferred(formatKb(bar.transfer)) : t.unknownSize
}

function formatKb(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)}MB`
  return `${Math.max(1, Math.round(bytes / 1024))}KB`
}

const REDUCED_QUERY = '(prefers-reduced-motion: reduce)'

function useMedia(query: string) {
  return useSyncExternalStore(
    (on) => {
      const q = window.matchMedia(query)
      q.addEventListener('change', on)
      return () => q.removeEventListener('change', on)
    },
    () => window.matchMedia(query).matches,
    () => false,
  )
}

const PHASE_COLOR: Record<PhaseKey, string> = {
  stalled: '#5c5c73',
  dns: '#2dd4bf',
  connect: '#a78bfa',
  tls: '#f472b6',
  ttfb: '#818cf8',
  download: '#f5f5fa',
}

function Detail({
  bar,
  start,
  onClose,
  t,
}: {
  bar: Bar
  start: number
  onClose: () => void
  t: TraceText
}) {
  const phases = bar.phases.filter((p) => p.ms >= 0.5)
  const total = phases.reduce((n, p) => n + p.ms, 0)
  const rows: [string, string][] = [
    [t.row.kind, `${t.kind[bar.kind]}${bar.self ? `, ${t.selfNote}` : ''}`],
    [t.row.start, `${Math.round(start)}ms`],
    [t.row.length, `${Math.round(bar.end - bar.start)}ms`],
    [t.row.transfer, transferText(bar, t)],
  ]
  if (bar.decoded) {
    rows.push([
      t.row.decoded,
      `${formatKb(bar.encoded)} → ${formatKb(bar.decoded)}`,
    ])
  }
  if (bar.protocol) rows.push([t.row.protocol, bar.protocol])
  if (bar.initiator) rows.push([t.row.initiator, bar.initiator])
  if (bar.blocking) {
    rows.push([t.row.blocking, bar.blocking === 'blocking' ? t.yes : t.no])
  }

  return (
    <aside
      className={aboutStyles.trace_detail}
      aria-label={t.detailOf(bar.label)}
    >
      <div className={aboutStyles.trace_detail_head}>
        <strong className={aboutStyles.trace_tooltip_title}>{bar.label}</strong>
        <button
          type="button"
          className={aboutStyles.trace_detail_close}
          onClick={onClose}
        >
          {t.close}
        </button>
      </div>
      <p className={aboutStyles.trace_detail_url}>{bar.url}</p>
      <dl className={aboutStyles.trace_detail_rows}>
        {rows.map(([k, v]) => (
          <div key={k} className={aboutStyles.trace_detail_row}>
            <dt>{k}</dt>
            <dd className={aboutStyles.trace_detail_value}>{v}</dd>
          </div>
        ))}
      </dl>
      {total > 0 ? (
        <>
          <div className={aboutStyles.trace_phase_bar}>
            {phases.map((p) => (
              <i
                key={p.key}
                style={{flexGrow: p.ms, background: PHASE_COLOR[p.key]}}
              />
            ))}
          </div>
          <ul className={aboutStyles.trace_phase_list}>
            {phases.map((p) => (
              <li key={p.key} className={aboutStyles.trace_legend_item}>
                <i
                  className={aboutStyles.trace_swatch}
                  style={{background: PHASE_COLOR[p.key]}}
                />
                {t.phase[p.key]}{' '}
                {p.ms < 10 ? p.ms.toFixed(1) : Math.round(p.ms)}ms
              </li>
            ))}
          </ul>
        </>
      ) : (
        <p className={aboutStyles.trace_detail_note}>{t.noPhases}</p>
      )}
    </aside>
  )
}

export default function LoadTrace() {
  const t = useTraceText()
  const [trace, setTrace] = useState<Trace | null>(null)
  const [replayKey, setReplayKey] = useState(0)
  const [visible, setVisible] = useState(true)
  const [hover, setHover] = useState<{i: number; x: number; y: number}>({
    i: -1,
    x: 0,
    y: 0,
  })
  const reduced = useMedia(REDUCED_QUERY)
  // 마우스가 있을 때만 휠과 Shift 조작을 안내한다
  const finePointer = useMedia('(pointer: fine)')
  const wrapRef = useRef<HTMLDivElement>(null)
  const labelEls = useRef<(HTMLDivElement | null)[]>([])
  const hudEl = useRef<HTMLDivElement>(null)
  const orbit = useRef<Orbit>({
    yaw: 0,
    pitch: 0,
    vYaw: 0,
    vPitch: 0,
    zoom: 1,
    pan: new THREE.Vector3(),
    panDx: 0,
    panDy: 0,
    dragging: false,
    panning: false,
    homing: false,
    touched: false,
    invalidate: () => {},
  })
  const [dragged, setDragged] = useState(false)
  const [grabbing, setGrabbing] = useState(false)
  const [focus, setFocus] = useState(-1)
  // 포인터 리스너는 한 번만 달아서 초점 상태를 ref로 본다
  const focusRef = useRef(focus)
  useEffect(() => {
    focusRef.current = focus
  }, [focus])
  // 측정이 끝나도 WebGL 초기화와 셰이더 컴파일이 남아 있다. 첫 프레임까지 로딩 표시를 유지한다.
  const [drawn, setDrawn] = useState(false)

  useEffect(() => {
    if (focus < 0) return undefined
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setFocus(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [focus])

  useEffect(() => {
    let alive = true
    void (async () => {
      const collected = await collectTrace(SELF_URL)
      if (alive) setTrace(collected)
    })()
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return undefined
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting))
    io.observe(el)
    return () => io.disconnect()
  }, [])

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return undefined
    let lastX = 0
    let lastY = 0
    let pressed = false
    // 눌려 있는 포인터별 마지막 위치. 둘이면 핀치(확대)와 두 손가락 끌기(이동)로 본다.
    const pts = new Map<number, {x: number; y: number}>()
    const touched = () => {
      const o = orbit.current
      if (!o.touched) {
        o.touched = true
        setDragged(true)
      }
      o.invalidate()
    }
    // 누르자마자 포인터를 잡으면 막대 클릭이 R3F에 닿지 않는다. 4px 넘게 움직여야 드래그로 본다.
    const down = (e: PointerEvent) => {
      if (e.button !== 0) return
      pts.set(e.pointerId, {x: e.clientX, y: e.clientY})
      pressed = true
      orbit.current.vYaw = 0
      orbit.current.vPitch = 0
      orbit.current.homing = false
      // Shift를 누른 채 끌면 돌리는 대신 옮긴다. 초점 모드에서는 보이지 않으니 받지 않는다.
      orbit.current.panning = e.shiftKey && focusRef.current < 0
      lastX = e.clientX
      lastY = e.clientY
      if (pts.size === 2 && !orbit.current.dragging) {
        orbit.current.dragging = true
        setGrabbing(true)
        setHover({i: -1, x: 0, y: 0})
      }
    }
    const move = (e: PointerEvent) => {
      const o = orbit.current
      const prev = pts.get(e.pointerId)
      if (prev && pts.size >= 2) {
        // 두 손가락 사이가 벌어진 비율만큼 당기고, 중점이 움직인 만큼 옮긴다
        const other = [...pts].find(([id]) => id !== e.pointerId)?.[1]
        if (other && focusRef.current < 0) {
          const before = Math.hypot(prev.x - other.x, prev.y - other.y)
          const after = Math.hypot(e.clientX - other.x, e.clientY - other.y)
          if (before > 0 && after > 0) {
            o.zoom = THREE.MathUtils.clamp(
              o.zoom * (before / after),
              ZOOM_MIN,
              ZOOM_MAX,
            )
          }
          o.panDx += (e.clientX - prev.x) / 2
          o.panDy += (e.clientY - prev.y) / 2
        }
        pts.set(e.pointerId, {x: e.clientX, y: e.clientY})
        touched()
        return
      }
      if (pressed && !o.dragging) {
        if (Math.hypot(e.clientX - lastX, e.clientY - lastY) < 4) return
        o.dragging = true
        el.setPointerCapture(e.pointerId)
        setGrabbing(true)
        setHover({i: -1, x: 0, y: 0})
      }
      if (!o.dragging) return
      if (o.panning) {
        o.panDx += e.clientX - lastX
        o.panDy += e.clientY - lastY
      } else {
        o.vYaw = (e.clientX - lastX) * 0.006
        // 터치의 세로 이동은 페이지 스크롤에 남겨 둔다
        o.vPitch = e.pointerType === 'mouse' ? (e.clientY - lastY) * -0.005 : 0
        o.yaw += o.vYaw
        o.pitch += o.vPitch
      }
      lastX = e.clientX
      lastY = e.clientY
      if (prev) pts.set(e.pointerId, {x: e.clientX, y: e.clientY})
      touched()
    }
    const up = (e: PointerEvent) => {
      pts.delete(e.pointerId)
      if (el.hasPointerCapture(e.pointerId))
        el.releasePointerCapture(e.pointerId)
      const rest = pts.values().next().value
      if (rest) {
        // 한 손가락이 남으면 그 손가락 기준으로 이어서 돌린다
        lastX = rest.x
        lastY = rest.y
        orbit.current.vYaw = 0
        orbit.current.vPitch = 0
        return
      }
      pressed = false
      orbit.current.dragging = false
      orbit.current.panning = false
      setGrabbing(false)
    }
    // touch-action이 pan-y라 두 손가락의 세로 이동을 브라우저가 스크롤로 가져갈 수 있다. 핀치 중에는 막는다.
    const touchMove = (e: TouchEvent) => {
      if (e.touches.length >= 2 && e.cancelable) e.preventDefault()
    }
    // 휠은 페이지 스크롤에 남기고, Ctrl/⌘를 누른 휠(트랙패드 핀치 포함)만 확대로 받는다
    const wheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey) || focusRef.current >= 0) return
      e.preventDefault()
      const o = orbit.current
      o.homing = false
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY
      o.zoom = THREE.MathUtils.clamp(
        o.zoom * Math.exp(dy * 0.002),
        ZOOM_MIN,
        ZOOM_MAX,
      )
      touched()
    }
    const home = () => {
      const o = orbit.current
      // 몇 바퀴 돌았어도 가까운 쪽으로 돌아온다
      o.yaw = Math.atan2(Math.sin(o.yaw), Math.cos(o.yaw))
      o.vYaw = 0
      o.vPitch = 0
      o.homing = true
      o.invalidate()
    }
    el.addEventListener('pointerdown', down)
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', up)
    el.addEventListener('pointercancel', up)
    el.addEventListener('touchmove', touchMove, {passive: false})
    el.addEventListener('wheel', wheel, {passive: false})
    el.addEventListener('dblclick', home)
    return () => {
      el.removeEventListener('pointerdown', down)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', up)
      el.removeEventListener('pointercancel', up)
      el.removeEventListener('touchmove', touchMove)
      el.removeEventListener('wheel', wheel)
      el.removeEventListener('dblclick', home)
    }
  }, [])

  const layout = useMemo(() => (trace ? layoutOf(trace) : null), [trace])

  const labels = useMemo<LabelSpec[]>(() => {
    if (!trace || !layout) return []
    const out: LabelSpec[] = trace.marks.map((m, i, all) => ({
      key: `mark-${m.label}`,
      // 두 표지가 가까우면 라벨 높이를 엇갈린다
      pos: new THREE.Vector3(
        layout.x(m.t),
        i > 0 && layout.u(m.t) - layout.u(all[i - 1].t) < 0.1 ? 1.5 : 1.2,
        -D / 2 - 0.4,
      ),
      text: `${m.label} ${Math.round(m.t)}ms`,
      tone: 'mark',
      at: layout.u(m.t),
    }))
    const self = layout.bars.find((b) => b.self)
    if (self) {
      out.push({
        key: 'self',
        pos: new THREE.Vector3(self.x1, self.h + 0.35, self.z),
        text: t.selfLabel,
        tone: 'self',
        at: layout.u(self.end),
      })
    }
    if (trace.longTasks.length) {
      out.push({
        key: 'main',
        pos: new THREE.Vector3(-W / 2, 0.05, layout.mainLaneZ),
        text: t.mainThread,
        tone: 'lane',
        at: 0,
      })
    }
    for (const ms of layout.ticks) {
      out.push({
        key: `tick-${ms}`,
        pos: new THREE.Vector3(layout.x(ms), 0, D / 2 + 0.55),
        text:
          ms >= 1000 ? `${(ms / 1000).toFixed(ms % 1000 ? 1 : 0)}s` : `${ms}ms`,
        tone: 'tick',
        at: layout.u(ms),
      })
    }
    for (const g of layout.gaps) {
      out.push({
        key: `gap-${g.t}`,
        pos: new THREE.Vector3(layout.x(g.t), 0, D / 2 + 0.55),
        text: t.gap(Math.round(g.ms)),
        tone: 'tick',
        at: layout.u(g.t),
      })
    }
    return out
  }, [trace, layout, t])

  const hovered = layout && hover.i >= 0 ? layout.bars[hover.i] : null
  const kinds = layout
    ? (Object.keys(KIND_COLOR) as BarKind[]).filter((k) =>
        layout.bars.some((b) => b.kind === k),
      )
    : []
  const fcp = trace?.marks.find((m) => m.label === 'FCP')
  const lcp = trace?.marks.find((m) => m.label === 'LCP')
  const summary = trace
    ? [
        t.resources(trace.bars.length),
        formatKb(trace.totalBytes),
        fcp && `FCP ${Math.round(fcp.t)}ms`,
        lcp && `LCP ${Math.round(lcp.t)}ms`,
      ]
        .filter(Boolean)
        .join(', ')
    : ''

  return (
    <figure className={aboutStyles.trace_figure}>
      <div
        ref={wrapRef}
        className={
          grabbing
            ? aboutStyles.trace_viewport_grabbing
            : aboutStyles.trace_viewport
        }
        // 같은 내용을 figcaption이 글로 전한다
        aria-hidden="true"
      >
        {!drawn && <TraceLoading />}
        {trace && layout && (
          <Canvas
            dpr={[1, 2]}
            frameloop={visible && !reduced ? 'always' : 'demand'}
            camera={{fov: FOV, near: 0.1, far: 80}}
            gl={{
              antialias: true,
              alpha: false,
              powerPreference: 'high-performance',
            }}
            onCreated={({gl}) => gl.setClearColor('#0a0a0f')}
            onPointerMissed={() => {
              setHover({i: -1, x: 0, y: 0})
              setFocus(-1)
            }}
            fallback={
              <p className={aboutStyles.trace_fallback}>
                {t.noWebgl} {summary}
              </p>
            }
          >
            <Scene
              trace={trace}
              layout={layout}
              replayKey={replayKey}
              reduced={reduced}
              labels={labels}
              labelEls={labelEls}
              hudEl={hudEl}
              hover={hover.i}
              orbit={orbit}
              focus={focus}
              onSelect={setFocus}
              onFirstFrame={() => setDrawn(true)}
              onHover={(i, e) => {
                const rect = wrapRef.current?.getBoundingClientRect()
                setHover({
                  i,
                  x: e && rect ? e.nativeEvent.clientX - rect.left : 0,
                  y: e && rect ? e.nativeEvent.clientY - rect.top : 0,
                })
              }}
            />
          </Canvas>
        )}
        <div className={aboutStyles.trace_overlay}>
          <div ref={hudEl} className={aboutStyles.trace_hud} />
          {labels.map((l, i) => (
            <div
              key={l.key}
              ref={(el) => {
                labelEls.current[i] = el
              }}
              className={aboutStyles.trace_label}
            >
              <span className={aboutStyles.trace_label_text[l.tone]}>
                {l.text}
              </span>
            </div>
          ))}
          <div className={aboutStyles.trace_bottom}>
            {trace && !dragged && (
              <p className={aboutStyles.trace_hint}>
                {finePointer ? t.hintMouse : t.hint}
              </p>
            )}
            <ul className={aboutStyles.trace_legend}>
              {kinds.map((k) => (
                <li key={k} className={aboutStyles.trace_legend_item}>
                  <i
                    className={aboutStyles.trace_swatch}
                    style={{background: KIND_COLOR[k]}}
                  />
                  {t.kind[k]}
                </li>
              ))}
              {trace?.longTasks.length ? (
                <li className={aboutStyles.trace_legend_item}>
                  <i
                    className={aboutStyles.trace_swatch}
                    style={{background: LONG_TASK_COLOR}}
                  />
                  {t.longTask}
                </li>
              ) : null}
            </ul>
          </div>
          {hovered && focus < 0 && (
            <div
              className={aboutStyles.trace_tooltip}
              style={{
                transform: `translate(${hover.x + 14}px, ${hover.y + 14}px)`,
              }}
            >
              <strong className={aboutStyles.trace_tooltip_title}>
                {hovered.label}
              </strong>
              <span>
                {t.kind[hovered.kind]}
                {hovered.self ? `, ${t.selfNote}` : ''}
              </span>
              <span>
                {t.timing(
                  Math.round(trace!.bars[hover.i].start),
                  Math.round(hovered.end - hovered.start),
                )}
              </span>
              <span>{transferText(hovered, t)}</span>
            </div>
          )}
        </div>
      </div>
      {layout && trace && focus >= 0 && (
        <Detail
          bar={layout.bars[focus]}
          start={trace.bars[focus].start}
          onClose={() => setFocus(-1)}
          t={t}
        />
      )}
      <TraceCaption
        stats={
          trace
            ? `${summary}${trace.hidden ? `, ${t.hidden(trace.hidden)}` : ''}`
            : t.reading
        }
        onReplay={
          trace
            ? () => {
                setFocus(-1)
                setReplayKey((k) => k + 1)
              }
            : undefined
        }
      />
    </figure>
  )
}

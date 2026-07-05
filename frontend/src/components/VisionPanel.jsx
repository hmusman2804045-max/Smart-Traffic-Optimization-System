import { motion, AnimatePresence } from 'framer-motion'

const CLASS_STYLE = {
  car: { color: '#00e5ff', w: 46, h: 30 },
  truck: { color: '#ff3df2', w: 56, h: 38 },
  bus: { color: '#ffb020', w: 60, h: 40 },
  motorcycle: { color: '#3dff8f', w: 26, h: 24 },
  person: { color: '#8b7bff', w: 16, h: 30 },
}

/* Fixed choreography: vehicles loop from the horizon toward the camera. */
const VEHICLES = [
  { cls: 'car', lane: -0.62, dur: 6.5, delay: 0, conf: 0.94 },
  { cls: 'truck', lane: -0.24, dur: 9, delay: 1.4, conf: 0.91 },
  { cls: 'car', lane: 0.22, dur: 5.6, delay: 2.8, conf: 0.97 },
  { cls: 'car', lane: 0.6, dur: 7.2, delay: 0.9, conf: 0.89 },
  { cls: 'motorcycle', lane: 0.02, dur: 4.8, delay: 3.6, conf: 0.86 },
  { cls: 'bus', lane: -0.45, dur: 10.5, delay: 5.2, conf: 0.95 },
  { cls: 'car', lane: 0.42, dur: 6.1, delay: 4.4, conf: 0.92 },
]

function Vehicle({ cls, lane, dur, delay, conf }) {
  const s = CLASS_STYLE[cls]
  // lane in [-1, 1]: converge toward centre at horizon, spread at bottom
  const xStart = 50 + lane * 7
  const xEnd = 50 + lane * 38
  return (
    <motion.div
      className="bbox"
      style={{ borderColor: s.color, width: s.w, height: s.h }}
      initial={{ left: `${xStart}%`, top: '30%', scale: 0.28, opacity: 0 }}
      animate={{
        left: [`${xStart}%`, `${xEnd}%`],
        top: ['30%', '96%'],
        scale: [0.28, 1.35],
        opacity: [0, 1, 1, 1, 0],
      }}
      transition={{ duration: dur, delay, repeat: Infinity, ease: [0.45, 0.05, 0.85, 0.6], repeatDelay: 0.4 }}
    >
      <span className="bbox-label" style={{ background: s.color }}>
        {cls} {conf.toFixed(2)}
      </span>
      {cls === 'person' ? (
        <div style={{
          position: 'absolute', inset: 3, borderRadius: '40% 40% 20% 20%',
          background: `linear-gradient(180deg, ${s.color}55, ${s.color}22)`,
        }} />
      ) : (
        <>
          <div style={{
            position: 'absolute', inset: 3, borderRadius: 5,
            background: `linear-gradient(180deg, ${s.color}40, #0a1220 75%)`,
            border: `1px solid ${s.color}66`,
          }} />
          <div style={{ position: 'absolute', bottom: 5, left: '18%', width: 5, height: 4, borderRadius: 2, background: '#fffbe6', boxShadow: '0 0 8px #fffbe6' }} />
          <div style={{ position: 'absolute', bottom: 5, right: '18%', width: 5, height: 4, borderRadius: 2, background: '#fffbe6', boxShadow: '0 0 8px #fffbe6' }} />
        </>
      )}
    </motion.div>
  )
}

function Pedestrian({ delay, top, dur, reverse }) {
  const s = CLASS_STYLE.person
  return (
    <motion.div
      className="bbox"
      style={{ borderColor: s.color, width: s.w, height: s.h, top }}
      animate={{ left: reverse ? ['86%', '6%'] : ['6%', '86%'], opacity: [0, 1, 1, 1, 0] }}
      transition={{ duration: dur, delay, repeat: Infinity, ease: 'linear', repeatDelay: 1.2 }}
    >
      <span className="bbox-label" style={{ background: s.color }}>person 0.88</span>
      <div style={{
        position: 'absolute', inset: 2, borderRadius: '45% 45% 22% 22%',
        background: `linear-gradient(180deg, ${s.color}66, ${s.color}1e)`,
      }} />
    </motion.div>
  )
}

function Road() {
  return (
    <svg viewBox="0 0 400 240" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
      <defs>
        <linearGradient id="roadFade" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#0d1626" stopOpacity="0.2" />
          <stop offset="1" stopColor="#111d33" stopOpacity="0.85" />
        </linearGradient>
      </defs>
      {/* horizon glow */}
      <rect x="0" y="58" width="400" height="3" fill="#00e5ff" opacity="0.18" />
      <polygon points="168,62 232,62 372,240 28,240" fill="url(#roadFade)" />
      {/* road edges */}
      <line x1="168" y1="62" x2="28" y2="240" stroke="#00e5ff" strokeWidth="1.4" opacity="0.55" />
      <line x1="232" y1="62" x2="372" y2="240" stroke="#00e5ff" strokeWidth="1.4" opacity="0.55" />
      {/* lane dividers */}
      <line x1="189" y1="62" x2="140" y2="240" stroke="#2b3d5e" strokeWidth="1.2" strokeDasharray="7 11" opacity="0.8" />
      <line x1="211" y1="62" x2="260" y2="240" stroke="#2b3d5e" strokeWidth="1.2" strokeDasharray="7 11" opacity="0.8" />
      <motion.line
        x1="200" y1="62" x2="200" y2="240"
        stroke="#00e5ff" strokeWidth="1.6" strokeDasharray="9 13" opacity="0.5"
        animate={{ strokeDashoffset: [0, -44] }}
        transition={{ duration: 1.1, repeat: Infinity, ease: 'linear' }}
      />
      {/* crosswalk */}
      {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
        <rect key={i} x={54 + i * 38} y={206} width={22} height={9} fill="#1d2c47" transform={`skewX(-18)`} />
      ))}
      {/* skyline blocks */}
      {[[10, 20, 26, 42], [48, 34, 20, 28], [330, 26, 24, 36], [362, 38, 22, 24], [86, 44, 16, 18], [300, 42, 18, 20]].map(([x, y, w, h], i) => (
        <rect key={i} x={x} y={y} width={w} height={h} fill="#0c1526" stroke="#1c3350" strokeWidth="0.8" />
      ))}
    </svg>
  )
}

const COUNT_KEYS = [
  ['cars', 'Cars'],
  ['trucks', 'Trucks'],
  ['buses', 'Buses'],
  ['motorcycles', 'Motos'],
  ['pedestrians', 'Peds'],
]

export default function VisionPanel({ counts, density, step }) {
  const total = counts.cars + counts.trucks + counts.buses + counts.motorcycles
  return (
    <motion.section
      className="panel"
      initial={{ y: 24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="panel-head">
        <div className="panel-title"><span className="dot c" /> Live Vision — YOLOv8</div>
        <div className="panel-sub">CAM-07 / SECTOR 4 INTERSECTION</div>
      </div>
      <div className="panel-body">
        <div className="cam">
          <Road />
          {VEHICLES.map((v, i) => <Vehicle key={i} {...v} />)}
          <Pedestrian delay={2} top="72%" dur={11} />
          <Pedestrian delay={8} top="78%" dur={13} reverse />
          <div className="cam-vignette" />
          <div className="cam-scanlines" />
          <div className="cam-hud">
            <span className="corner tl" /><span className="corner tr" />
            <span className="corner bl" /><span className="corner br" />
            <div className="cam-id">CAM-07 · 1920×1080 · 30FPS</div>
            <div className="cam-rec">
              <motion.span
                style={{ width: 7, height: 7, borderRadius: '50%', background: '#ff2d55', display: 'inline-block' }}
                animate={{ opacity: [1, 0.15, 1] }}
                transition={{ duration: 1.4, repeat: Infinity }}
              />
              REC
            </div>
            <div className="cam-meta">
              INFERENCE 11.2ms · FRAME {String(1400 + step * 66).padStart(6, '0')} · {total} OBJ TRACKED
            </div>
          </div>
        </div>

        <div className="cam-counts">
          {COUNT_KEYS.map(([key, label]) => (
            <div className="count-cell" key={key}>
              <AnimatePresence mode="popLayout">
                <motion.div
                  className="v"
                  key={counts[key]}
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  exit={{ y: -10, opacity: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  {counts[key]}
                </motion.div>
              </AnimatePresence>
              <div className="k">{label}</div>
            </div>
          ))}
        </div>
      </div>
    </motion.section>
  )
}

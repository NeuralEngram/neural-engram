// Small dependency-free SVG charts.

const W = 560
const H = 200
const PAD = { l: 36, r: 12, t: 12, b: 28 }

function scale(v, d0, d1, r0, r1) {
  return d1 === d0 ? r0 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)
}

function Axes({ xTicks, yTicks = [0, 0.5, 1] }) {
  return (
    <g className="axes">
      {yTicks.map((t) => {
        const y = scale(t, 0, 1, H - PAD.b, PAD.t)
        return (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y} y2={y} className="grid" />
            <text x={PAD.l - 6} y={y + 4} textAnchor="end">{Math.round(t * 100)}%</text>
          </g>
        )
      })}
      {xTicks.map(({ x, label }, i) => (
        <text key={label} x={x} y={H - 8}
          textAnchor={xTicks.length > 1 && i === xTicks.length - 1 ? 'end'
            : i === 0 && xTicks.length > 1 ? 'start' : 'middle'}>
          {label}
        </text>
      ))}
    </g>
  )
}

// Daily average retention for one person.
export function TrendChart({ points }) {
  if (!points?.length) {
    return <p className="muted small">A trend appears after a day or two of use.</p>
  }
  const n = points.length
  const x = (i) => scale(i, 0, Math.max(1, n - 1), PAD.l + 8, W - PAD.r - 8)
  const y = (v) => scale(v, 0, 1, H - PAD.b, PAD.t)
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${x(i)},${y(p.avg_retention)}`).join(' ')
  const label = (d) => new Date(d + 'T00:00:00')
    .toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
  const ticks = n === 1
    ? [{ x: x(0), label: label(points[0].day) }]
    : [{ x: x(0), label: label(points[0].day) },
      { x: x(n - 1), label: label(points[n - 1].day) }]
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`Average retention over ${n} day${n > 1 ? 's' : ''}`}>
      <Axes xTicks={ticks} />
      <path d={path} className="line accent" />
      {points.map((p, i) => (
        <circle key={p.day} cx={x(i)} cy={y(p.avg_retention)} r="3.5" className="dot">
          <title>{`${label(p.day)}: ${Math.round(p.avg_retention * 100)}%`}</title>
        </circle>
      ))}
    </svg>
  )
}

const TONES = ['line-1', 'line-2', 'line-3', 'line-4']

// Forgetting curves: standard memory vs each Alzheimer's stage.
export function CompareChart({ curves }) {
  if (!curves?.length) return null
  const maxH = curves[0].points[curves[0].points.length - 1].hours
  const x = (h) => scale(h, 0, maxH, PAD.l, W - PAD.r)
  const y = (v) => scale(v, 0, 1, H - PAD.b, PAD.t)
  const ticks = [0, 24, 48, 72, 96, 120].filter((h) => h <= maxH)
    .map((h) => ({ x: x(h), label: `${h} h` }))
  return (
    <figure className="chart-figure">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label="How fast memories fade for each memory profile">
        <Axes xTicks={ticks} />
        {curves.map((c, i) => (
          <path key={c.label} className={`line ${TONES[i % TONES.length]}`}
            d={c.points.map((p, j) => `${j ? 'L' : 'M'}${x(p.hours)},${y(p.retention)}`).join(' ')} />
        ))}
      </svg>
      <figcaption className="legend">
        {curves.map((c, i) => (
          <span key={c.label}>
            <i className={`swatch ${TONES[i % TONES.length]}`} />
            {c.label}{c.multiplier !== 1 ? ` (${c.multiplier}× faster)` : ''}
          </span>
        ))}
      </figcaption>
    </figure>
  )
}

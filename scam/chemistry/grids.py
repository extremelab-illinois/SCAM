# SPDX-License-Identifier: MIT
def parse_number_grid(text: str, *, name: str) -> list[float]:
    value = text.strip()
    if not value:
        raise ValueError(f"Empty {name} grid")

    if ":" in value and "," not in value:
        parts = [float(p) for p in value.split(":")]
        if len(parts) == 2:
            return parts
        if len(parts) != 3:
            raise ValueError(f"{name} range must be start:stop:step or start:stop")

        start, stop, step = parts
        if step == 0.0:
            raise ValueError(f"{name} step cannot be zero")
        if (stop - start) * step < 0.0:
            raise ValueError(f"{name} step sign inconsistent with start/stop")

        values = []
        current = start
        tol = 1.0e-12 * max(1.0, abs(stop))

        while (current <= stop + tol) if step > 0.0 else (current >= stop - tol):
            values.append(current)
            current += step

        if abs(values[-1] - stop) > 1.0e-9 * max(1.0, abs(stop)):
            values.append(stop)

        return values

    values = [float(p.strip()) for p in value.split(",") if p.strip()]
    if not values:
        raise ValueError(f"Could not parse {name}: {text!r}")

    return values

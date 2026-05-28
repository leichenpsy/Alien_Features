import numpy as np
import matplotlib.pyplot as plt


def wrap_angle_deg(angle):
    """
    Wrap angle to [0, 360).
    """
    return angle % 360.0


def angular_distance_deg(a, b):
    """
    Compute the smallest circular angular distance between two angles.
    """
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def quantize_angle(angle, unit=0.1, wrap=True):
    """
    Quantize an angle or value to the nearest unit.
    """
    value = round(angle / unit) * unit
    value = round(value, 10)

    if wrap:
        value = wrap_angle_deg(value)

    return value


def sample_von_mises_deg(mean_deg, kappa, rng=None):
    """
    Sample one angle in degrees from a von Mises distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    mean_rad = np.deg2rad(mean_deg)
    sample_rad = rng.vonmises(mean_rad, kappa)

    return wrap_angle_deg(np.rad2deg(sample_rad))


def circular_span_deg(points):
    """
    Compute the smallest circular arc span containing all points.
    """
    points = np.sort(np.asarray(points, dtype=float) % 360.0)

    if len(points) <= 1:
        return 0.0

    gaps = np.diff(points)
    wrap_gap = 360.0 - points[-1] + points[0]
    gaps = np.append(gaps, wrap_gap)

    largest_gap = np.max(gaps)

    return 360.0 - largest_gap


def sample_vonmises_with_min_spacing(
    center_deg,
    count,
    kappa,
    existing_points=None,
    min_dist_deg=3.0,
    unit=0.1,
    rng=None,
    max_tries=10000,
):
    """
    Sample angles around a center using a von Mises distribution,
    while enforcing minimum circular spacing from existing points.
    """
    if rng is None:
        rng = np.random.default_rng()

    if existing_points is None:
        existing_points = []

    accepted = []
    center_rad = np.deg2rad(center_deg)

    for _ in range(count):
        found = False

        for _try in range(max_tries):
            angle_rad = rng.vonmises(center_rad, kappa)
            angle_deg = quantize_angle(np.rad2deg(angle_rad), unit=unit)

            all_previous = existing_points + accepted

            if all(
                angular_distance_deg(angle_deg, prev) >= min_dist_deg
                for prev in all_previous
            ):
                accepted.append(angle_deg)
                found = True
                break

        if not found:
            raise RuntimeError(
                f"Could not sample {count} points around {center_deg}° "
                f"with min_dist_deg={min_dist_deg} and kappa={kappa}."
            )

    return accepted


def generate_circle_point_sets_internal(
    set_labels,
    base_means_deg,
    n,
    kappa=80.0,
    side_offset_deg=8.0,
    mean_jitter_deg=3.0,
    unit=0.1,
    rng=None,
    min_within_pairwise_dist_deg=3.0,
):
    """
    Generate circular point sets with internal metadata.

    This function is intended for internal validation only.

    Returns
    -------
    dict
        Internal sample containing:
        - set labels
        - all generated points
        - left/right points
        - jittered centers
        - left/right centers
    """
    if rng is None:
        rng = np.random.default_rng()

    base_means_deg = np.asarray(base_means_deg, dtype=float)

    if len(base_means_deg) != len(set_labels):
        raise ValueError("base_means_deg and set_labels must have the same length.")

    if n % 2 != 0:
        raise ValueError("n must be even so that half the points are left and half are right.")

    generated_sets = {}
    generated_info = {}

    for label, base_mean in zip(set_labels, base_means_deg):
        # Random jitter for this set center
        mean_jitter = rng.uniform(-mean_jitter_deg, mean_jitter_deg)
        mean_jitter = quantize_angle(mean_jitter, unit=unit, wrap=False)

        # Jittered center mean
        jittered_mean = quantize_angle(
            base_mean + mean_jitter,
            unit=unit,
            wrap=True,
        )

        # Left and right centers
        left_center = quantize_angle(
            jittered_mean - side_offset_deg,
            unit=unit,
            wrap=True,
        )

        right_center = quantize_angle(
            jittered_mean + side_offset_deg,
            unit=unit,
            wrap=True,
        )

        left_points = sample_vonmises_with_min_spacing(
            center_deg=left_center,
            count=n // 2,
            kappa=kappa,
            existing_points=[],
            min_dist_deg=min_within_pairwise_dist_deg,
            unit=unit,
            rng=rng,
            max_tries=10000,
        )

        right_points = sample_vonmises_with_min_spacing(
            center_deg=right_center,
            count=n - n // 2,
            kappa=kappa,
            existing_points=left_points,
            min_dist_deg=min_within_pairwise_dist_deg,
            unit=unit,
            rng=rng,
            max_tries=10000,
        )

        points = left_points + right_points

        generated_sets[label] = {
            "all_points": points,
            "left_points": left_points,
            "right_points": right_points,
        }

        generated_info[label] = {
            "base_mean_deg": float(base_mean),
            "mean_jitter_deg": mean_jitter,
            "jittered_mean_deg": jittered_mean,
            "left_center_deg": left_center,
            "right_center_deg": right_center,
        }

    internal_sample = {
        "set_labels": list(set_labels),
        "sets": generated_sets,
        "generated_info": generated_info,
        "parameters": {
            "base_means_deg": base_means_deg.tolist(),
            "n_per_set": n,
            "kappa": kappa,
            "side_offset_deg": side_offset_deg,
            "mean_jitter_deg": mean_jitter_deg,
            "unit": unit,
            "min_within_pairwise_dist_deg": min_within_pairwise_dist_deg,
        },
    }

    return internal_sample

def check_circle_point_sets_internal(
    internal_sample,
    min_within_pairwise_dist_deg=1.8,
    max_within_span_deg=45.0,
    min_between_set_dist_deg=30.0,
    min_center_dist_deg=70.0,
    verbose=False,
):
    """
    Check whether generated point sets meet requirements.

    This function expects the internal sample structure.
    """
    set_labels = internal_sample["set_labels"]
    sets = internal_sample["sets"]
    info = internal_sample["generated_info"]

    # 1. Check within-set pairwise distances
    for label in set_labels:
        points = sets[label]["all_points"]

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = angular_distance_deg(points[i], points[j])

                if dist < min_within_pairwise_dist_deg:
                    if verbose:
                        print(
                            f"Failed within-set pairwise distance for {label}: "
                            f"points {points[i]} and {points[j]}, "
                            f"distance={dist:.2f}, "
                            f"required>={min_within_pairwise_dist_deg}"
                        )
                    return False

        # 2. Check within-set span
        span = circular_span_deg(points)

        if span > max_within_span_deg:
            if verbose:
                print(
                    f"Failed within-set span for {label}: "
                    f"span={span:.2f}, "
                    f"max allowed={max_within_span_deg}"
                )
            return False

    # 3. Check between-set point separation
    for a in range(len(set_labels)):
        for b in range(a + 1, len(set_labels)):
            label_a = set_labels[a]
            label_b = set_labels[b]

            points_a = sets[label_a]["all_points"]
            points_b = sets[label_b]["all_points"]

            for p_a in points_a:
                for p_b in points_b:
                    dist = angular_distance_deg(p_a, p_b)

                    if dist < min_between_set_dist_deg:
                        if verbose:
                            print(
                                f"Failed between-set distance for {label_a} and {label_b}: "
                                f"points {p_a} and {p_b}, "
                                f"distance={dist:.2f}, "
                                f"required>={min_between_set_dist_deg}"
                            )
                        return False

    # 4. Check between-center distances
    for a in range(len(set_labels)):
        for b in range(a + 1, len(set_labels)):
            label_a = set_labels[a]
            label_b = set_labels[b]

            center_a = info[label_a]["jittered_mean_deg"]
            center_b = info[label_b]["jittered_mean_deg"]

            dist = angular_distance_deg(center_a, center_b)

            if dist < min_center_dist_deg:
                if verbose:
                    print(
                        f"Failed center distance for {label_a} and {label_b}: "
                        f"centers {center_a} and {center_b}, "
                        f"distance={dist:.2f}, "
                        f"required>={min_center_dist_deg}"
                    )
                return False

    return True

def simplify_sample(internal_sample):
    """
    Convert an internal sample into the simple public return format.

    Returns
    -------
    dict
        Dictionary with labels as keys and generated point lists as values.
    """
    return {
        label: internal_sample["sets"][label]["all_points"]
        for label in internal_sample["set_labels"]
    }

def generate_until_valid(
    set_labels,
    base_means_deg,
    n=8,
    kappa=45.0,
    side_offset_deg=8.0,
    mean_jitter_deg=3.0,
    unit=0.1,
    min_within_pairwise_dist_deg=3.0,
    max_within_span_deg=70.0,
    min_between_set_dist_deg=20.0,
    min_center_dist_deg=60.0,
    max_attempts=100000,
    rng=None,
    verbose=False,
):
    """
    Generate samples repeatedly until all requirements are met.

    Internally, this uses metadata such as jittered centers for validation.
    Publicly, it returns only a simple dictionary:

    {
        label_1: [points...],
        label_2: [points...],
        ...
    }

    Returns
    -------
    dict
        Simple sample dictionary with labels as keys and point lists as values.
    """
    if rng is None:
        rng = np.random.default_rng()

    for attempt in range(1, max_attempts + 1):
        internal_sample = generate_circle_point_sets_internal(
            set_labels=set_labels,
            base_means_deg=base_means_deg,
            n=n,
            kappa=kappa,
            side_offset_deg=side_offset_deg,
            mean_jitter_deg=mean_jitter_deg,
            unit=unit,
            rng=rng,
            min_within_pairwise_dist_deg=min_within_pairwise_dist_deg,
        )

        is_valid = check_circle_point_sets_internal(
            internal_sample=internal_sample,
            min_within_pairwise_dist_deg=min_within_pairwise_dist_deg,
            max_within_span_deg=max_within_span_deg,
            min_between_set_dist_deg=min_between_set_dist_deg,
            min_center_dist_deg=min_center_dist_deg,
            verbose=verbose,
        )

        if is_valid:
            return simplify_sample(internal_sample)

    raise RuntimeError(
        f"Could not generate a valid sample after {max_attempts} attempts. "
        f"Try relaxing constraints, increasing kappa, or reducing n."
    )


def pol2cart_deg(angle_deg, radius=1.0):
    """
    Convert polar coordinates to Cartesian coordinates.
    """
    angle_rad = np.deg2rad(angle_deg)
    x = radius * np.cos(angle_rad)
    y = radius * np.sin(angle_rad)
    return x, y


def plot_circle_point_sets(
    sample,
    title=None,
    point_radius=1.0,
    show_labels=True,
    figsize=(7, 7),
):
    """
    Plot a simple sample dictionary.

    Parameters
    ----------
    sample : dict
        Dictionary with labels as keys and point lists as values.

        Example:
        {
            "red": [350.1, 355.2, 5.8, 10.3],
            "yellow": [82.2, 86.5, 96.4, 101.7],
        }
    """
    set_labels = list(sample.keys())

    default_colors = ["red", "gold", "green", "blue", "purple", "orange"]

    colors = {
        label: default_colors[i % len(default_colors)]
        for i, label in enumerate(set_labels)
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Draw main circle
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.5)

    # Draw axes
    ax.axhline(0, color="lightgray", linewidth=1)
    ax.axvline(0, color="lightgray", linewidth=1)

    # Add reference degree labels
    reference_angles = [0, 90, 180, 270]
    reference_labels = ["0°", "90°", "180°", "270°"]

    for angle, label in zip(reference_angles, reference_labels):
        x, y = pol2cart_deg(angle, radius=1.15)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Plot sampled points
    for label in set_labels:
        color = colors[label]
        points = sample[label]

        for angle in points:
            x, y = pol2cart_deg(angle, radius=point_radius)

            ax.scatter(
                x,
                y,
                s=35,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                marker="o",
                zorder=4,
            )

            if show_labels:
                lx, ly = pol2cart_deg(angle, radius=1.08)
                ax.text(
                    lx,
                    ly,
                    f"{angle:.1f}°",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    # Legend
    for label in set_labels:
        ax.scatter(
            [],
            [],
            s=35,
            color=colors[label],
            edgecolor="black",
            label=label,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

    ax.set_xticks([])
    ax.set_yticks([])

    if title is None:
        title = "Circular sampled points"

    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=8)

    plt.tight_layout()

    return fig

sample = generate_until_valid(['red','yellow','green','blue'],[0, 90, 180, 270], n = 8,
    kappa=80.0,
    side_offset_deg=10.0,
    mean_jitter_deg=2,
    unit=0.1,
    min_within_pairwise_dist_deg=4,
    max_within_span_deg=70.0,
    min_between_set_dist_deg=30.0,
    min_center_dist_deg=60.0)
fig = plot_circle_point_sets(sample)
plt.show()
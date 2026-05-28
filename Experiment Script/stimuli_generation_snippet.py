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

    Parameters
    ----------
    a : float
        First angle in degrees.

    b : float
        Second angle in degrees.

    Returns
    -------
    float
        Angular distance in degrees, in [0, 180].
    """
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def quantize_angle(angle, unit=0.1, wrap=True):
    """
    Quantize an angle or value to the nearest unit.

    Parameters
    ----------
    angle : float
        Angle or value to quantize.

    unit : float
        Quantization unit. For example, unit=0.1 means one decimal place.

    wrap : bool
        If True, wrap the result to [0, 360).
        If False, do not wrap.

    Returns
    -------
    float
        Quantized value.
    """
    value = round(angle / unit) * unit

    # Avoid floating point artifacts like 25.200000000000003
    value = round(value, 10)

    if wrap:
        value = wrap_angle_deg(value)

    return value


def sample_von_mises_deg(mean_deg, kappa, rng=None):
    """
    Sample one angle in degrees from a von Mises distribution.

    Parameters
    ----------
    mean_deg : float
        Mean direction in degrees.

    kappa : float
        Concentration parameter.
        Larger kappa means samples are more tightly concentrated around mean_deg.

    rng : np.random.Generator or None
        NumPy random generator.

    Returns
    -------
    float
        Sampled angle in degrees, wrapped to [0, 360).
    """
    if rng is None:
        rng = np.random.default_rng()

    mean_rad = np.deg2rad(mean_deg)
    sample_rad = rng.vonmises(mean_rad, kappa)

    return wrap_angle_deg(np.rad2deg(sample_rad))


def circular_span_deg(points):
    """
    Compute the smallest circular arc span containing all points.

    Example
    -------
    Points [350, 10] have circular span 20 degrees, not 340 degrees.

    Parameters
    ----------
    points : array-like
        Angles in degrees.

    Returns
    -------
    float
        Minimal circular span in degrees.
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

    Parameters
    ----------
    center_deg : float
        Center angle in degrees.

    count : int
        Number of new points to sample.

    kappa : float
        Von Mises concentration.

    existing_points : list or None
        Points already accepted in the same set.

    min_dist_deg : float
        Minimum allowed angular distance between points.

    unit : float
        Quantization unit in degrees.

    rng : np.random.Generator
        Random number generator.

    max_tries : int
        Maximum attempts for each point.

    Returns
    -------
    list
        Accepted sampled angles.
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

def generate_circle_point_sets(
    set_labels,
    base_means_deg,
    n,
    kappa=80.0,
    side_offset_deg=8.0,
    mean_jitter_deg=3.0,
    unit=0.1,
    rng=None,
    min_within_pairwise_dist_deg = 3
):
    """
    Generate four sets of sampled points on a circle.

    Each set has a base mean. A random jitter is added to the base mean.
    Then half of the points are sampled around a left center and half around
    a right center.

    For one set:
        mean_jitter   ~ Uniform(-mean_jitter_deg, mean_jitter_deg)
        jittered_mean = base_mean + mean_jitter
        left_center   = jittered_mean - side_offset_deg
        right_center  = jittered_mean + side_offset_deg

    Points are sampled from von Mises distributions centered at left_center
    and right_center.

    Parameters
    ----------
    set_labels : list of str
        Labels for the four sets.
        Example: ["set_A", "set_B", "set_C", "set_D"]

    base_means_deg : list or array-like
        Base center means for each set in degrees.
        Example: [0, 90, 180, 270]

    n : int
        Number of points per set. Must be even.

    kappa : float
        Concentration parameter for the von Mises distribution.
        Larger values produce tighter clusters.

    side_offset_deg : float
        Offset from the jittered mean to create left and right centers.

    mean_jitter_deg : float
        Maximum absolute jitter added to each base mean.
        Jitter is sampled from Uniform(-mean_jitter_deg, mean_jitter_deg).

    unit : float
        Quantization unit in degrees.
        Default 0.1 allows values like 25.2 but not 25.23.

    rng : np.random.Generator or None
        NumPy random generator.

    Returns
    -------
    dict
        Dictionary containing parameter values and generated point sets.
    """
    if rng is None:
        rng = np.random.default_rng()

    base_means_deg = np.asarray(base_means_deg, dtype=float)

    if len(base_means_deg) != 4:
        raise ValueError("base_means_deg must contain exactly four values.")

    if len(set_labels) != 4:
        raise ValueError("set_labels must contain exactly four labels.")

    if n % 2 != 0:
        raise ValueError("n must be even so that half the points are left and half are right.")

    generated_sets = {}
    generated_info = {}

    half_n = n // 2

    for label, base_mean in zip(set_labels, base_means_deg):
        # Random jitter for this set center
        mean_jitter = rng.uniform(-mean_jitter_deg, mean_jitter_deg)
        mean_jitter = quantize_angle(mean_jitter, unit=unit, wrap=False)

        # Jittered center mean
        jittered_mean = quantize_angle(base_mean + mean_jitter, unit=unit, wrap=True)

        # Left and right centers
        left_center = quantize_angle(jittered_mean - side_offset_deg, unit=unit, wrap=True)
        right_center = quantize_angle(jittered_mean + side_offset_deg, unit=unit, wrap=True)

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

    result = {

        "parameters": {
            "base_means_deg": base_means_deg.tolist(),
            "n_per_set": n,
            "kappa": kappa,
            "side_offset_deg": side_offset_deg,
            "mean_jitter_deg": mean_jitter_deg,
            "unit": unit,
        },

        "set_labels": list(set_labels),

        "generated_info": generated_info,

        "sets": generated_sets,
    }

    return result


def check_circle_point_sets(
    sample,
    min_within_pairwise_dist_deg=1.8,
    max_within_span_deg=45.0,
    min_between_set_dist_deg=30.0,
    min_center_dist_deg=70.0,
    verbose=False,
):
    """
    Check whether generated point sets meet perceptual requirements.

    Requirements
    ------------
    1. Within each set, no two points should be too close.
    2. Within each set, points should not spread too widely.
    3. Between different sets, points should be clearly separated.
    4. Jittered centers of different sets should be clearly separated.

    Parameters
    ----------
    sample : dict
        Output dictionary from generate_circle_point_sets.

    min_within_pairwise_dist_deg : float
        Minimum angular distance between any two points in the same set.

    max_within_span_deg : float
        Maximum allowed circular span of each set.

    min_between_set_dist_deg : float
        Minimum angular distance between points from different sets.

    min_center_dist_deg : float
        Minimum angular distance between jittered means of different sets.

    verbose : bool
        If True, print the reason when a check fails.

    Returns
    -------
    bool
        True if all checks pass, False otherwise.
    """
    set_labels = sample["set_labels"]
    sets = sample["sets"]
    info = sample["generated_info"]

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


def generate_until_valid(
    set_labels,
    base_means_deg,
    n = 8,
    kappa=45.0,
    side_offset_deg=8.0,
    mean_jitter_deg=3.0,
    unit=0.1,
    min_within_pairwise_dist_deg=3,
    max_within_span_deg=70.0,
    min_between_set_dist_deg=20.0,
    min_center_dist_deg=60.0,
    max_attempts=100000,
    rng=None,
    verbose=False,
):
    """
    Generate samples repeatedly until all requirements are met.

    Parameters
    ----------


    set_labels : list of str
        Labels for the four sets.

    base_means_deg : list or array-like
        Base means in degrees.

    n : int
        Number of points per set. Must be even.

    kappa : float
        Von Mises concentration parameter.

    side_offset_deg : float
        Offset from jittered mean to left and right centers.

    mean_jitter_deg : float
        Jitter range. Jitter is sampled from
        Uniform(-mean_jitter_deg, mean_jitter_deg).

    unit : float
        Quantization unit in degrees.

    min_within_pairwise_dist_deg : float
        Minimum pairwise distance within each set.

    max_within_span_deg : float
        Maximum circular span within each set.

    min_between_set_dist_deg : float
        Minimum point-to-point distance between sets.

    min_center_dist_deg : float
        Minimum distance between jittered means.

    max_attempts : int
        Maximum number of generation attempts.

    rng : np.random.Generator or None
        NumPy random generator.

    verbose : bool
        If True, print failure reasons during checking.

    Returns
    -------
    dict
        Valid sample dictionary with number of attempts included.

    Raises
    ------
    RuntimeError
        If no valid sample is found within max_attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    for attempt in range(1, max_attempts + 1):
        sample = generate_circle_point_sets(
            set_labels=set_labels,
            base_means_deg=base_means_deg,
            n=n,
            kappa=kappa,
            side_offset_deg=side_offset_deg,
            mean_jitter_deg=mean_jitter_deg,
            unit=unit,
            rng=rng,
        )

        is_valid = check_circle_point_sets(
            sample=sample,
            min_within_pairwise_dist_deg=min_within_pairwise_dist_deg,
            max_within_span_deg=max_within_span_deg,
            min_between_set_dist_deg=min_between_set_dist_deg,
            min_center_dist_deg=min_center_dist_deg,
            verbose=verbose,
        )

        if is_valid:
            sample["validity_requirements"] = {
                "min_within_pairwise_dist_deg": min_within_pairwise_dist_deg,
                "max_within_span_deg": max_within_span_deg,
                "min_between_set_dist_deg": min_between_set_dist_deg,
                "min_center_dist_deg": min_center_dist_deg,
            }

            sample["generation_status"] = {
                "is_valid": True,
                "attempts": attempt,
            }
            print(sample['sets'])
            return sample

    raise RuntimeError(
        f"Could not generate a valid sample after {max_attempts} attempts. "
        f"Try relaxing constraints, increasing kappa, or reducing n."
    )


def pol2cart_deg(angle_deg, radius=1.0):
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    angle_deg : float
        Angle in degrees.

    radius : float
        Radius.

    Returns
    -------
    tuple
        x, y Cartesian coordinates.
    """
    angle_rad = np.deg2rad(angle_deg)
    x = radius * np.cos(angle_rad)
    y = radius * np.sin(angle_rad)
    return x, y


def plot_circle_point_sets(
    sample,
    title=None,
    point_radius=1.0,
    center_radius=0.82,
    show_centers=True,
    show_left_right_centers=True,
    show_labels=True,
    figsize=(7, 7),
):
    """
    Plot generated circular point sets on a unit circle.

    Parameters
    ----------
    sample : dict
        Output from generate_circle_point_sets or generate_until_valid.

    title : str or None
        Plot title.

    point_radius : float
        Radius where sampled points are drawn.

    center_radius : float
        Radius where centers are drawn.

    show_centers : bool
        Whether to show jittered mean centers.

    show_left_right_centers : bool
        Whether to show left/right sub-centers.

    show_labels : bool
        Whether to label points by set.

    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object.
    """
    set_labels = sample["set_labels"]
    sets = sample["sets"]
    info = sample["generated_info"]

    colors = {
        set_labels[0]: "red",
        set_labels[1]: "gold",
        set_labels[2]: "green",
        set_labels[3]: "blue",
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
    reference_labels = ["0° red", "90° yellow", "180° green", "270° blue"]

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
        all_points = sets[label]["all_points"]
        left_points = sets[label]["left_points"]
        right_points = sets[label]["right_points"]

        # Plot left points
        for idx, angle in enumerate(left_points):
            x, y = pol2cart_deg(angle, radius=point_radius)
            ax.scatter(
                x,
                y,
                s=25,
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

        # Plot right points
        for idx, angle in enumerate(right_points):
            x, y = pol2cart_deg(angle, radius=point_radius)
            ax.scatter(
                x,
                y,
                s=25,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                marker="s",
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

        # Plot jittered center
        if show_centers:
            center = info[label]["jittered_mean_deg"]
            cx, cy = pol2cart_deg(center, radius=center_radius)
            ax.scatter(
                cx,
                cy,
                s=160,
                color=color,
                edgecolor="black",
                linewidth=1.2,
                marker="*",
                zorder=5,
            )

        # Plot left/right sub-centers
        if show_left_right_centers:
            left_center = info[label]["left_center_deg"]
            right_center = info[label]["right_center_deg"]

            lx, ly = pol2cart_deg(left_center, radius=center_radius)
            rx, ry = pol2cart_deg(right_center, radius=center_radius)

            ax.scatter(
                lx,
                ly,
                s=25,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                marker="o",
                alpha=0.5,
                zorder=3,
            )

            ax.scatter(
                rx,
                ry,
                s=25,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                marker="s",
                alpha=0.5,
                zorder=3,
            )

    # Legend proxy artists
    for label in set_labels:
        ax.scatter(
            [],
            [],
            s=25,
            color=colors[label],
            edgecolor="black",
            label=label,
        )

    ax.scatter(
        [],
        [],
        s=25,
        color="white",
        edgecolor="black",
        marker="o",
        label="left sampled points",
    )

    ax.scatter(
        [],
        [],
        s=25,
        color="white",
        edgecolor="black",
        marker="s",
        label="right sampled points",
    )

    ax.scatter(
        [],
        [],
        s=140,
        color="white",
        edgecolor="black",
        marker="*",
        label="jittered center",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

    ax.set_xticks([])
    ax.set_yticks([])

    if title is None:
        title = sample.get("Circular sampled points")

    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=8)

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
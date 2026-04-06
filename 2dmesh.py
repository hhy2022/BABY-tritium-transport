import gmsh


def generate_baby_upper_2d_mesh(
    fname="baby_2d.msh",
    show_gui=True,
):
    gmsh.initialize()
    gmsh.model.add("BABY_upper_2D_OpenMC_based")
    occ = gmsh.model.occ

    # =========================================================
    # Radial dimensions [cm]
    # =========================================================

    r_axis = 0.0
    r_heater = 0.439
    r_cllif = 7.000
    r_inconel = 7.300

    # =========================================================
    # Axial thicknesses [cm]
    # =========================================================

    t_base = 0.786
    t_alumina = 0.635
    t_he = 0.600
    t_inconel = 0.300
    t_heater_gap = 0.878
    t_cllif = 6.388 + 0.13022
    t_gap = 4.605
    t_cap = 1.422

    # =========================================================
    # Axial coordinates (absolute positions, cm)
    # =========================================================

    y0 = 0.0  # bottom of base

    # Layer interfaces (following OpenMC z_plane definitions)
    y1 = y0 + t_base
    y2 = y1 + t_alumina
    y3 = y2 + t_he
    y4 = y3 + t_inconel
    y5 = y4 + t_cllif
    y6 = y5 + t_gap
    y7 = y6 + t_cap

    t_heater = y7 - (y4 + t_heater_gap)

    # Heater
    y_heater_bottom = y4 + t_heater_gap
    y_heater_top = y_heater_bottom + t_heater

    # Base
    y_base_bottom = y0

    # Alumina
    y_alumina_top = y2

    # Helium layer
    y_he_bottom = y2
    y_he_top = y3

    # Inconel bottom
    y_inconel_top = y4

    # CLLiF
    y_cllif_bottom = y4
    y_cllif_top = y5

    # Gap
    y_gap_bottom = y5
    y_gap_top = y6

    # Cap
    y_IV_top = y7

    # =========================================================
    # Geometry containers
    # =========================================================

    surfaces = {}
    # ---------------------------------------------------------
    # Inconel 625 cap: bottom part
    # ---------------------------------------------------------
    surfaces["IV_bottom"] = occ.addRectangle(r_axis, y_he_top, 0, r_inconel, t_inconel)

    # ---------------------------------------------------------
    # Heater
    # The heater starts inside the CLLiF region and extends upward
    # beyond the vessel top.
    # ---------------------------------------------------------
    surfaces["heater"] = occ.addRectangle(
        r_axis, y_heater_bottom, 0, r_heater, t_heater
    )

    # ---------------------------------------------------------
    # CLLiF natural
    # ---------------------------------------------------------
    surfaces["cllif_lower"] = occ.addRectangle(
        r_axis, y_cllif_bottom, 0, r_cllif, y_heater_bottom - y_cllif_bottom
    )

    surfaces["cllif_upper"] = occ.addRectangle(
        r_heater, y_heater_bottom, 0, r_cllif - r_heater, y_cllif_top - y_heater_bottom
    )

    # ---------------------------------------------------------
    # Inconel 625 cap: cylindrical wall around the inner vessel
    # ---------------------------------------------------------
    surfaces["IV_wall"] = occ.addRectangle(
        r_cllif, y_inconel_top, 0, r_inconel - r_cllif, y_gap_top - y_inconel_top
    )

    # ---------------------------------------------------------
    # Inconel 625 cap: top part
    # ---------------------------------------------------------
    surfaces["IV_top"] = occ.addRectangle(
        r_heater, y_gap_top, 0, r_inconel - r_heater, t_cap
    )

    occ.synchronize()

    # =========================================================
    # Helium region
    # Helium exists only inside the vessel, from the top of alumina
    # to the top of the high section, after subtracting:
    # - cap regions
    # - CLLiF
    # - heater
    # =========================================================

    helium_box_out = occ.addRectangle(
        r_axis, y_alumina_top, 0, r_inconel, y7 - y_alumina_top
    )
    occ.synchronize()

    helium_tools_out = [
        (2, surfaces["IV_bottom"]),
        (2, surfaces["IV_wall"]),
        (2, surfaces["IV_top"]),
        # (2, surfaces["firebrick"]),
        (2, surfaces["heater"]),
    ]

    helium_cut, _ = occ.cut(
        [(2, helium_box_out)],
        helium_tools_out,
        removeObject=True,
        removeTool=False,
    )
    occ.synchronize()

    helium_inner = occ.addRectangle(
        r_heater, y_cllif_top, 0, r_cllif - r_heater, y_gap_top - y_cllif_top
    )
    occ.synchronize()

    helium_cut, _ = occ.cut(
        helium_cut,
        [(2, helium_inner)],
        removeObject=True,
        removeTool=False,
    )
    occ.synchronize()

    helium_tags = [tag for dim, tag in helium_cut if dim == 2]

    # =========================================================
    # Fragment all regions so interfaces are conformal
    # =========================================================

    all_entities = []
    for tag in surfaces.values():
        all_entities.append((2, tag))
    all_entities.extend((2, tag) for tag in helium_tags)

    occ.fragment(all_entities, [])
    occ.synchronize()

    # =========================================================
    # Reclassify surfaces after fragmentation
    # =========================================================

    physical_surfaces = {
        "inconel625": [],
        "alumina": [],
        "helium": [],
        "cllif_natural": [],
        "heater": [],
        "firebrick": [],
    }

    tol = 1e-8

    def in_range(value, lower, upper):
        return (lower - tol) <= value <= (upper + tol)

    for dim, tag in gmsh.model.getEntities(2):
        xc, yc, _ = occ.getCenterOfMass(dim, tag)

        # -----------------------------------------------------
        # Heater
        # -----------------------------------------------------
        if in_range(xc, r_axis, r_heater) and in_range(
            yc, y_heater_bottom, y_heater_top
        ):
            physical_surfaces["heater"].append(tag)
            continue

        # # -----------------------------------------------------
        # # Alumina
        # # -----------------------------------------------------
        # if in_range(yc, y_alumina_bottom, y_alumina_top) and in_range(
        #     xc, r_axis, r_vessel
        # ):
        #     physical_surfaces["alumina"].append(tag)
        #     continue

        # # -----------------------------------------------------
        # # Firebrick
        # # -----------------------------------------------------
        # if in_range(xc, r_he, r_firebrick) and in_range(
        #     yc, y_alumina_top, y_firebrick_top
        # ):
        #     physical_surfaces["firebrick"].append(tag)
        #     continue

        # -----------------------------------------------------
        # CLLiF natural
        # -----------------------------------------------------
        if in_range(yc, y_cllif_bottom, y_heater_bottom) and in_range(
            xc, r_axis, r_cllif
        ):
            physical_surfaces["cllif_natural"].append(tag)
            continue

        if in_range(yc, y_heater_bottom, y_cllif_top) and in_range(
            xc, r_heater, r_cllif
        ):
            physical_surfaces["cllif_natural"].append(tag)
            continue

        # -----------------------------------------------------
        # Inconel 625
        # -----------------------------------------------------
        # if in_range(yc, y_base_bottom, y_base_top) and in_range(xc, r_axis, r_external):
        #     physical_surfaces["inconel625"].append(tag)
        #     continue

        # if in_range(xc, r_vessel, r_external) and in_range(yc, y_base_top, y_high_top):
        #     physical_surfaces["inconel625"].append(tag)
        #     continue

        if in_range(yc, y_he_bottom, y_inconel_top) and in_range(xc, r_axis, r_inconel):
            physical_surfaces["inconel625"].append(tag)
            continue

        if in_range(xc, r_cllif, r_inconel) and in_range(yc, y_inconel_top, y_gap_top):
            physical_surfaces["inconel625"].append(tag)
            continue

        if in_range(yc, y_gap_bottom, y_IV_top) and in_range(xc, r_heater, r_inconel):
            physical_surfaces["inconel625"].append(tag)
            continue

        # if in_range(yc, y_high_top, y_cover_top) and in_range(xc, r_heater, r_external):
        #     physical_surfaces["inconel625"].append(tag)
        #     continue

        # -----------------------------------------------------
        # Remaining internal void is helium
        # -----------------------------------------------------
        # if in_range(yc, y_alumina_top, y_high_top) and in_range(xc, r_axis, r_vessel):
        #     physical_surfaces["helium"].append(tag)
        #     continue

    # Remove duplicates while preserving order
    for key, tags in physical_surfaces.items():
        seen = set()
        deduped = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                deduped.append(tag)
        physical_surfaces[key] = deduped

    # =========================================================
    # Create physical groups
    # =========================================================

    physical_ids = {
        "inconel625": 1,
        "alumina": 2,
        "helium": 3,
        "cllif_natural": 4,
        "heater": 5,
        "firebrick": 6,
    }

    for name, group_id in physical_ids.items():
        tags = physical_surfaces[name]
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, group_id)
            gmsh.model.setPhysicalName(2, group_id, name)

    # =========================================================
    # Boundary groups
    # =========================================================

    boundary_lines = gmsh.model.getBoundary(
        [(2, tag) for tags in physical_surfaces.values() for tag in tags],
        oriented=False,
        recursive=False,
    )
    boundary_lines = list({tag for dim, tag in boundary_lines if dim == 1})

    axis_lines = []
    outer_lines = []
    bottom_lines = []
    heater_tip_lines = []

    for line in boundary_lines:
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, line)

        if abs(xmin - r_axis) < 1e-9 and abs(xmax - r_axis) < 1e-9:
            axis_lines.append(line)

        # if abs(xmin - r_external) < 1e-9 and abs(xmax - r_external) < 1e-9:
        #     outer_lines.append(line)

        if abs(ymin - y_base_bottom) < 1e-9 and abs(ymax - y_base_bottom) < 1e-9:
            bottom_lines.append(line)

        if abs(ymin - y_heater_top) < 1e-9 and abs(ymax - y_heater_top) < 1e-9:
            heater_tip_lines.append(line)

    if axis_lines:
        gmsh.model.addPhysicalGroup(1, axis_lines, 101)
        gmsh.model.setPhysicalName(1, 101, "axis")

    if outer_lines:
        gmsh.model.addPhysicalGroup(1, outer_lines, 102)
        gmsh.model.setPhysicalName(1, 102, "outer_boundary")

    if bottom_lines:
        gmsh.model.addPhysicalGroup(1, bottom_lines, 103)
        gmsh.model.setPhysicalName(1, 103, "bottom_boundary")

    if heater_tip_lines:
        gmsh.model.addPhysicalGroup(1, heater_tip_lines, 104)
        gmsh.model.setPhysicalName(1, 104, "heater_top_boundary")

    # =========================================================
    # Mesh
    # =========================================================

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.20)
    gmsh.model.mesh.generate(2)
    gmsh.write(fname)

    if show_gui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    generate_baby_upper_2d_mesh()

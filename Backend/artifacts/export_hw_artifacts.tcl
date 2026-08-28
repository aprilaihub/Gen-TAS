# ============================================================
# Export Vivado hardware artifacts for PYNQ / board testing
#
# Default use:
#   tclsh Backend/artifacts/export_hw_artifacts.tcl
#
# Override examples:
#   tclsh export_hw_artifacts.tcl CONFIG_FILE=/path/to/vivado_config.tcl DESIGN_NAME=lightcnn_a
#   tclsh export_hw_artifacts.tcl DESIGN_NAME=lightcnn_a PROJECT_DIR=/path/to/project
# ============================================================

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set CONFIG [dict create]

# Defaults for the LightCNN full A+B+C build.
dict set CONFIG DESIGN_NAME "lightcnn_all"
dict set CONFIG PROJECT_NAME "lightcnn_all_vivado"
dict set CONFIG PROJECT_DIR [file join $SCRIPT_DIR "lightcnn_all_vivado"]
dict set CONFIG BD_NAME "design_1"
dict set CONFIG EXPORT_ROOT [file join $SCRIPT_DIR "hardware_exports"]
dict set CONFIG CONFIG_FILE ""

# Preserve command-line values so they take precedence over values loaded from
# the shared Vivado configuration file.
set ARG_OVERRIDES [dict create]
foreach arg $argv {
    if {[regexp {([^=]+)=(.*)} $arg -> key value]} {
        dict set ARG_OVERRIDES $key $value
    }
}

if {[dict exists $ARG_OVERRIDES CONFIG_FILE]} {
    dict set CONFIG CONFIG_FILE [dict get $ARG_OVERRIDES CONFIG_FILE]
}

if {[dict get $CONFIG CONFIG_FILE] ne ""} {
    set config_file [file normalize [dict get $CONFIG CONFIG_FILE]]
    if {![file exists $config_file]} {
        error "Vivado CONFIG_FILE does not exist: $config_file"
    }
    source $config_file
}

dict for {key value} $ARG_OVERRIDES {
    dict set CONFIG $key $value
}

if {![dict exists $CONFIG TOP_NAME] || [dict get $CONFIG TOP_NAME] eq ""} {
    dict set CONFIG TOP_NAME "[dict get $CONFIG BD_NAME]_wrapper"
}

proc C {key} {
    global CONFIG
    return [dict get $CONFIG $key]
}

set bit_src [file join [C PROJECT_DIR] "[C PROJECT_NAME].runs" "impl_1" "[C TOP_NAME].bit"]
set hwh_src [file join [C PROJECT_DIR] "[C PROJECT_NAME].gen" "sources_1" "bd" [C BD_NAME] "hw_handoff" "[C BD_NAME].hwh"]

set export_dir [file join [C EXPORT_ROOT] [C DESIGN_NAME]]
set bit_dst [file join $export_dir "[C DESIGN_NAME].bit"]
set hwh_dst [file join $export_dir "[C DESIGN_NAME].hwh"]

if {[dict exists $CONFIG REPORT_DIR] && [C REPORT_DIR] ne ""} {
    set report_src_dir [C REPORT_DIR]
} else {
    set report_src_dir [file join [C PROJECT_DIR] reports]
}
set report_dst_dir [file join $export_dir reports]

foreach src [list $bit_src $hwh_src] {
    if {![file exists $src]} {
        error "Required hardware artifact does not exist: $src"
    }
}

file mkdir $export_dir
file copy -force $bit_src $bit_dst
file copy -force $hwh_src $hwh_dst

set exported_reports {}
foreach report_name {module_utilization.csv power_report.csv timing_paths.csv} {
    set report_src [file join $report_src_dir $report_name]

    if {[file exists $report_src]} {
        file mkdir $report_dst_dir
        set report_dst [file join $report_dst_dir $report_name]
        file copy -force $report_src $report_dst
        lappend exported_reports $report_dst
    }
}

puts "============================================================"
puts "Exported hardware artifacts"
puts "Design: [C DESIGN_NAME]"
puts "Output directory: $export_dir"
puts "BIT: $bit_dst"
puts "HWH: $hwh_dst"
foreach report $exported_reports {
    puts "REPORT: $report"
}
puts "============================================================"

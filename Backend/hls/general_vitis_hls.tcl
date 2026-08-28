# ============================================================
# Generic Vitis HLS Project Builder
# Supports:
#   - config-driven project creation
#   - source/testbench file lists or automatic source discovery
#   - C simulation, C synthesis, RTL co-simulation, and IP export
# ============================================================

# ============================================================
# Default configuration
# ============================================================

if {![info exists CONFIG]} {
    set CONFIG [dict create]
}

proc config_default {key value} {
    global CONFIG
    if {![dict exists $CONFIG $key]} {
        dict set CONFIG $key $value
    }
}

# Project / solution
config_default PROJECT_NAME "lightcnn_all_hls"
config_default PROJECT_DIR  "[pwd]/build/hls"
config_default SOLUTION_NAME "solution1"

# Device / clock
config_default PART "xczu7ev-ffvc1156-2-e"
config_default CLOCK_PERIOD_NS 10.0
config_default CLOCK_UNCERTAINTY_NS 1.25

# HLS top and sources
config_default TOP_FUNCTION "lightcnn_top"
config_default SOURCE_DIR "[pwd]/Backend/examples/lightcnn/lightcnn_sep"
config_default SOURCE_FILES ""
config_default TB_FILES ""
config_default SOURCE_CFLAGS ""
config_default TB_CFLAGS ""

# Build controls
config_default RUN_CSIM   0
config_default RUN_CSYNTH 1
config_default RUN_COSIM  0
config_default RUN_EXPORT 1

# Export controls
config_default EXPORT_FORMAT "ip_catalog"
config_default EXPORT_FLOW   "impl"
config_default RTL_LANGUAGE  "verilog"
config_default IP_VERSION    "1.0"
config_default IP_DISPLAY_NAME ""
config_default IP_DESCRIPTION  ""

# Optional external config file. It should set CONFIG entries using:
#   dict set CONFIG KEY VALUE
config_default CONFIG_FILE ""

# ============================================================
# Override CONFIG from command line
#
# Examples:
#   vitis_hls -f general_vitis_hls.tcl CONFIG_FILE=/path/to/lightcnn_all_hls.tcl
#   vitis_hls -f general_vitis_hls.tcl TOP_FUNCTION=lightcnn_top RUN_COSIM=1
# ============================================================

set cli_overrides {}
foreach arg $argv {
    if {[regexp {([^=]+)=(.*)} $arg -> key value]} {
        dict set CONFIG $key $value
        if {$key ne "CONFIG_FILE"} {
            lappend cli_overrides [list $key $value]
        }
    } elseif {[file exists $arg]} {
        dict set CONFIG CONFIG_FILE $arg
    }
}

if {[dict get $CONFIG CONFIG_FILE] ne ""} {
    source [dict get $CONFIG CONFIG_FILE]
}

# Explicit command-line values take precedence over values in the config file.
foreach override $cli_overrides {
    dict set CONFIG [lindex $override 0] [lindex $override 1]
}

# ============================================================
# Helper procedures
# ============================================================

proc C {key} {
    global CONFIG
    return [dict get $CONFIG $key]
}

proc config_has {key} {
    global CONFIG
    return [dict exists $CONFIG $key]
}

proc normalize_list {value} {
    if {$value eq ""} {
        return {}
    }
    return $value
}

proc discover_source_files {source_dir} {
    set source_files {}
    set tb_files {}

    foreach pattern {*.c *.cc *.cpp *.cxx} {
        foreach path [glob -nocomplain -directory $source_dir $pattern] {
            set name [string tolower [file tail $path]]
            if {[regexp {(tb|test)} $name]} {
                lappend tb_files $path
            } else {
                lappend source_files $path
            }
        }
    }

    return [list $source_files $tb_files]
}

proc hls_project_path {path project_dir} {
    set abs_path [file normalize $path]
    set abs_project_dir [file normalize $project_dir]
    set project_parent [file dirname $abs_project_dir]

    if {[string first "${abs_project_dir}/" $abs_path] == 0} {
        return [string range $abs_path [expr {[string length $abs_project_dir] + 1}] end]
    }

    if {[string first "${project_parent}/" $abs_path] == 0} {
        set suffix [string range $abs_path [expr {[string length $project_parent] + 1}] end]
        return [file join ".." $suffix]
    }

    return $abs_path
}

proc add_hls_files {files mode project_dir cflags} {
    foreach src $files {
        if {![file exists $src]} {
            error "Missing HLS source file: $src"
        }

        set hls_src [hls_project_path $src $project_dir]

        if {$mode eq "tb"} {
            if {$cflags eq ""} {
                add_files -tb $hls_src
            } else {
                add_files -tb -cflags $cflags $hls_src
            }
        } else {
            if {$cflags eq ""} {
                add_files $hls_src
            } else {
                add_files -cflags $cflags $hls_src
            }
        }
    }
}

# ============================================================
# Derived data
# ============================================================

set project_dir [C PROJECT_DIR]
set source_dir  [C SOURCE_DIR]
set source_files [normalize_list [C SOURCE_FILES]]
set tb_files     [normalize_list [C TB_FILES]]

if {![file exists $source_dir]} {
    error "SOURCE_DIR does not exist: $source_dir"
}

if {[llength $source_files] == 0 && [llength $tb_files] == 0} {
    set discovered [discover_source_files $source_dir]
    set source_files [lindex $discovered 0]
    set tb_files     [lindex $discovered 1]
}

if {[llength $source_files] == 0} {
    error "No HLS design source files found. Set SOURCE_FILES or check SOURCE_DIR."
}

set ip_display_name [C IP_DISPLAY_NAME]
if {$ip_display_name eq ""} {
    set ip_display_name "[C TOP_FUNCTION]_IP"
}

set ip_description [C IP_DESCRIPTION]
if {$ip_description eq ""} {
    set ip_description "Vitis HLS-generated IP for [C TOP_FUNCTION]"
}

# ============================================================
# Build flow
# ============================================================

puts "============================================================"
puts "Running Vitis HLS build"
puts "Project:      $project_dir"
puts "Solution:     [C SOLUTION_NAME]"
puts "Top function: [C TOP_FUNCTION]"
puts "Part:         [C PART]"
puts "Clock:        [C CLOCK_PERIOD_NS] ns"
puts "Source dir:   $source_dir"
puts "============================================================"

set original_dir [pwd]
file mkdir [file dirname $project_dir]
open_project -reset $project_dir
cd $project_dir

add_hls_files $source_files "src" $project_dir [C SOURCE_CFLAGS]
add_hls_files $tb_files "tb" $project_dir [C TB_CFLAGS]

set_top [C TOP_FUNCTION]

open_solution -reset [C SOLUTION_NAME] -flow_target vivado
set_part [C PART]
create_clock -period [C CLOCK_PERIOD_NS] -name default

if {[C CLOCK_UNCERTAINTY_NS] ne ""} {
    set_clock_uncertainty [C CLOCK_UNCERTAINTY_NS]
}

if {[C RUN_CSIM]} {
    puts "Running C simulation..."
    csim_design
}

if {[C RUN_CSYNTH]} {
    puts "Running C synthesis..."
    csynth_design
}

if {[C RUN_COSIM]} {
    puts "Running RTL co-simulation..."
    cosim_design
}

if {[C RUN_EXPORT]} {
    puts "Exporting RTL IP..."
    export_design \
        -format [C EXPORT_FORMAT] \
        -rtl [C RTL_LANGUAGE] \
        -flow [C EXPORT_FLOW] \
        -version [C IP_VERSION] \
        -description $ip_description \
        -display_name $ip_display_name
}

close_project
cd $original_dir

puts "============================================================"
puts "Vitis HLS build completed successfully."
puts "Project:  $project_dir"
puts "Solution: [C SOLUTION_NAME]"
puts "============================================================"

# Vitis HLS 2024.1 can remain at its interactive prompt after a script passed
# with -f.  Return control to the pipeline wrapper once the build is complete.
exit 0

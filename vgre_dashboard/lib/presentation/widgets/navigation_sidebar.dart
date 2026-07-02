import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../../application/telemetry_bloc.dart';
import '../../core/theme/vgre_theme.dart';
import '../../core/docs_links.dart';

class VgreNavigationSidebar extends StatefulWidget {
  final int selectedIndex;
  final Function(int) onDestinationSelected;
  final bool isCompact;

  const VgreNavigationSidebar({
    super.key,
    required this.selectedIndex,
    required this.onDestinationSelected,
    this.isCompact = false,
  });

  @override
  State<VgreNavigationSidebar> createState() => _VgreNavigationSidebarState();
}

class _VgreNavigationSidebarState extends State<VgreNavigationSidebar> {
  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      width: widget.isCompact ? 80 : 280,
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.02),
        border: const Border(
          right: BorderSide(color: Colors.white12, width: 1),
        ),
      ),
      child: Column(
        crossAxisAlignment:
            widget.isCompact ? CrossAxisAlignment.center : CrossAxisAlignment.start,
        children: [
          // Logo Section
          Padding(
            padding: EdgeInsets.fromLTRB(
              widget.isCompact ? 0 : 32.0,
              16.0,
              widget.isCompact ? 0 : 32.0,
              32.0,
            ),
            child: widget.isCompact
                ? _compactLogo()
                : _fullLogo(),
          ),

          const SizedBox(height: 16),

          // Navigation Items
          Expanded(
            child: ListView(
              padding: EdgeInsets.symmetric(horizontal: widget.isCompact ? 8 : 16),
              children: [
                if (!widget.isCompact) _navHeader("MAIN"),
                _NavItem(
                  icon: Icons.dashboard_outlined,
                  label: "Dashboard",
                  index: 0,
                  isCompact: widget.isCompact,
                  selectedIndex: widget.selectedIndex,
                  onTap: () => widget.onDestinationSelected(0),
                ),
                _NavItem(
                  icon: Icons.code,
                  label: "Kernel Explorer",
                  index: 1,
                  isCompact: widget.isCompact,
                  selectedIndex: widget.selectedIndex,
                  onTap: () => widget.onDestinationSelected(1),
                ),
                _NavItem(
                  icon: Icons.hub_outlined,
                  label: "Cluster Topology",
                  index: 2,
                  isCompact: widget.isCompact,
                  selectedIndex: widget.selectedIndex,
                  onTap: () => widget.onDestinationSelected(2),
                ),
                const SizedBox(height: 16),
                if (!widget.isCompact) _navHeader("TUNING"),
                _NavItem(
                  icon: Icons.tune,
                  label: "Hardware Tuning",
                  index: 3,
                  isCompact: widget.isCompact,
                  selectedIndex: widget.selectedIndex,
                  onTap: () => widget.onDestinationSelected(3),
                ),
                const SizedBox(height: 16),
                if (!widget.isCompact) _navHeader("SYSTEM"),
                _NavItem(
                  icon: Icons.storage_outlined,
                  label: "Memory Analysis",
                  index: 4,
                  isCompact: widget.isCompact,
                  selectedIndex: widget.selectedIndex,
                  onTap: () => widget.onDestinationSelected(4),
                ),
                const SizedBox(height: 16),
                if (!widget.isCompact) _navHeader("ADMIN"),
                _NavItem(
                  icon: Icons.settings_outlined,
                  label: "Dashboard Settings",
                  index: 5,
                  isCompact: widget.isCompact,
                  selectedIndex: widget.selectedIndex,
                  onTap: () => widget.onDestinationSelected(5),
                ),
                const SizedBox(height: 16),
                if (!widget.isCompact) _navHeader("HELP & DOCS"),
                // These open the documentation website in the default browser
                // rather than switching the in-app page (index stays -1 so none
                // ever highlights as the active page).
                _ExternalNavItem(
                  icon: Icons.menu_book_outlined,
                  label: "Documentation",
                  url: DocsLinks.home,
                  isCompact: widget.isCompact,
                ),
                _ExternalNavItem(
                  icon: Icons.rocket_launch_outlined,
                  label: "Quick Start",
                  url: DocsLinks.quickStart,
                  isCompact: widget.isCompact,
                ),
                _ExternalNavItem(
                  icon: Icons.api_outlined,
                  label: "API Reference",
                  url: DocsLinks.apiReference,
                  isCompact: widget.isCompact,
                ),
                _ExternalNavItem(
                  icon: Icons.code_off_outlined,
                  label: "Source on GitHub",
                  url: DocsLinks.github,
                  isCompact: widget.isCompact,
                ),
                const SizedBox(height: 16),
              ],
            ),
          ),

          // Footer Section
          if (!widget.isCompact)
            Padding(
              padding: const EdgeInsets.all(24.0),
              child: BlocBuilder<TelemetryBloc, TelemetryState>(
                builder: (context, state) {
                  final String version = (state is TelemetryActive) ? state.backendVersion : "v0.0.0";
                  final int nodeCount = (state is TelemetryActive) ? state.telemetry.clusterNodes.length : 0;
                  
                  return Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.03),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: Colors.white10),
                    ),
                    child: Column(
                      children: [
                        const _ConnectionStatusBadge(),
                        const SizedBox(height: 12),
                        const Divider(color: Colors.white10, height: 1),
                        const SizedBox(height: 12),
                        _footerItem("RUNTIME", "v$version"),
                        const SizedBox(height: 12),
                        _footerItem("NODES", "$nodeCount CONNECTED"),
                      ],
                    ),
                  );
                },
              ),
            ),
        ],
      ),
    );
  }

  Widget _compactLogo() {
    return Container(
      width: 44,
      height: 44,
      decoration: BoxDecoration(
        color: VgreTheme.primaryNeon.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: VgreTheme.primaryNeon.withValues(alpha: 0.5),
        ),
        boxShadow: [
          BoxShadow(
            color: VgreTheme.primaryNeon.withValues(alpha: 0.2),
            blurRadius: 10,
          ),
        ],
      ),
      child: const Icon(
        Icons.memory,
        color: VgreTheme.primaryNeon,
        size: 24,
      ),
    );
  }

  Widget _fullLogo() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: VgreTheme.primaryNeon.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: VgreTheme.primaryNeon.withValues(alpha: 0.5),
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: VgreTheme.primaryNeon.withValues(alpha: 0.2),
                      blurRadius: 10,
                    ),
                  ],
                ),
                child: const Icon(
                  Icons.memory,
                  color: VgreTheme.primaryNeon,
                  size: 24,
                ),
              ),
              const SizedBox(width: 16),
              Flexible(
                child: Text(
                  "VGRE / X",
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.orbitron(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 2,
                    color: Colors.white,
                  ),
                ),
              ),
            ],
          ),
        const SizedBox(height: 8),
        Text(
          "VIRTUAL GPU RUNTIME",
          style: GoogleFonts.inter(
            fontSize: 10,
            fontWeight: FontWeight.w500,
            letterSpacing: 1.5,
            color: VgreTheme.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _navHeader(String label) {
    return Padding(
      padding: const EdgeInsets.only(left: 16, bottom: 12),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          color: VgreTheme.textMuted.withValues(alpha: 0.7),
          letterSpacing: 2,
        ),
      ),
    );
  }

  Widget _footerItem(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 9,
            fontWeight: FontWeight.bold,
            color: VgreTheme.textMuted,
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            fontSize: 9,
            fontWeight: FontWeight.bold,
            color: Colors.white70,
          ),
        ),
      ],
    );
  }
}

class _NavItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final int index;
  final int selectedIndex;
  final bool isCompact;
  final VoidCallback onTap;

  const _NavItem({
    required this.icon,
    required this.label,
    required this.index,
    required this.selectedIndex,
    required this.isCompact,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final bool isActive = index == selectedIndex;

    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          hoverColor: VgreTheme.primaryNeon.withValues(alpha: 0.05),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            padding: EdgeInsets.symmetric(
              horizontal: isActive ? 16 : 16,
              vertical: 12,
            ),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              color: isActive
                  ? VgreTheme.primaryNeon.withValues(alpha: 0.08)
                  : Colors.transparent,
              border: Border.all(
                color: isActive
                    ? VgreTheme.primaryNeon.withValues(alpha: 0.2)
                    : Colors.transparent,
              ),
            ),
            child: Row(
              mainAxisAlignment:
                  isCompact ? MainAxisAlignment.center : MainAxisAlignment.start,
              children: [
                Icon(
                  icon,
                  size: 20,
                  color: isActive ? VgreTheme.primaryNeon : VgreTheme.textMuted,
                ),
                if (!isCompact) ...[
                  const SizedBox(width: 16),
                  Expanded(
                    child: Text(
                      label,
                      style: GoogleFonts.inter(
                        fontSize: 14,
                        fontWeight: isActive ? FontWeight.w600 : FontWeight.w500,
                        color: isActive ? Colors.white : VgreTheme.textMuted,
                        letterSpacing: 0.2,
                      ),
                    ),
                  ),
                  if (isActive)
                    Container(
                      width: 6,
                      height: 6,
                      decoration: BoxDecoration(
                        color: VgreTheme.primaryNeon,
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: VgreTheme.primaryNeon.withValues(alpha: 0.5),
                            blurRadius: 10,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                    ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// A sidebar entry that opens a documentation URL in the user's browser instead
/// of switching the in-app page. Visually matches [_NavItem] (never "active")
/// and adds an external-link affordance; on failure it shows a snackbar with the
/// URL so the user can still reach it.
class _ExternalNavItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final String url;
  final bool isCompact;

  const _ExternalNavItem({
    required this.icon,
    required this.label,
    required this.url,
    required this.isCompact,
  });

  Future<void> _open(BuildContext context) async {
    final ok = await DocsLinks.open(url);
    if (!ok && context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: VgreTheme.surface,
          content: Text(
            'Could not open browser. Docs: $url',
            style: const TextStyle(color: Colors.white),
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () => _open(context),
          borderRadius: BorderRadius.circular(12),
          hoverColor: VgreTheme.primaryNeon.withValues(alpha: 0.05),
          child: Tooltip(
            message: isCompact ? label : url,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                mainAxisAlignment:
                    isCompact ? MainAxisAlignment.center : MainAxisAlignment.start,
                children: [
                  Icon(icon, size: 20, color: VgreTheme.textMuted),
                  if (!isCompact) ...[
                    const SizedBox(width: 16),
                    Expanded(
                      child: Text(
                        label,
                        style: GoogleFonts.inter(
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                          color: VgreTheme.textMuted,
                          letterSpacing: 0.2,
                        ),
                      ),
                    ),
                    Icon(
                      Icons.north_east,
                      size: 14,
                      color: VgreTheme.textMuted.withValues(alpha: 0.6),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Live/stale backend-connection indicator. Ticks every second and compares the
/// last successful telemetry poll (TelemetryActive.lastUpdated) to now, so the
/// UI never presents stale data as live: LIVE (fresh), DELAYED, or STALE.
class _ConnectionStatusBadge extends StatefulWidget {
  const _ConnectionStatusBadge();

  @override
  State<_ConnectionStatusBadge> createState() => _ConnectionStatusBadgeState();
}

class _ConnectionStatusBadgeState extends State<_ConnectionStatusBadge> {
  Timer? _tick;

  @override
  void initState() {
    super.initState();
    _tick = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _tick?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final state = context.read<TelemetryBloc>().state;
    final DateTime? last =
        (state is TelemetryActive) ? state.lastUpdated : null;

    String label;
    Color color;
    if (last == null) {
      label = "CONNECTING";
      color = VgreTheme.neonBlue;
    } else {
      final ageMs = DateTime.now().difference(last).inMilliseconds;
      if (ageMs <= 2000) {
        label = "LIVE";
        color = VgreTheme.neonGreen;
      } else if (ageMs <= 6000) {
        label = "DELAYED";
        color = const Color(0xFFFFB020);
      } else {
        label = "STALE";
        color = VgreTheme.neonRed;
      }
    }

    return Row(
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
            boxShadow: [BoxShadow(color: color.withValues(alpha: 0.6), blurRadius: 8)],
          ),
        ),
        const SizedBox(width: 8),
        Text(
          label,
          style: GoogleFonts.jetBrainsMono(
            color: color,
            fontSize: 11,
            fontWeight: FontWeight.w700,
            letterSpacing: 1.2,
          ),
        ),
        const Spacer(),
        Text(
          "BACKEND",
          style: GoogleFonts.jetBrainsMono(
            color: VgreTheme.textMuted,
            fontSize: 9,
            letterSpacing: 1.5,
          ),
        ),
      ],
    );
  }
}

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../core/theme/vgre_theme.dart';

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
            padding: EdgeInsets.symmetric(
              vertical: 32.0,
              horizontal: widget.isCompact ? 0 : 32.0,
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
              ],
            ),
          ),

          // Footer Section
          if (!widget.isCompact)
            Padding(
              padding: const EdgeInsets.all(24.0),
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.03),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.white10),
                ),
                child: Column(
                  children: [
                    _footerItem("RUNTIME", "v2.1.0-ALPHA"),
                    const SizedBox(height: 12),
                    _footerItem("NODES", "3 CONNECTED"),
                  ],
                ),
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

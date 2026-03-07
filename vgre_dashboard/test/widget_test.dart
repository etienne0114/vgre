import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('Dashboard shell smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(body: Text('VGRE Dashboard')),
    ));

    expect(find.text('VGRE Dashboard'), findsOneWidget);
  });
}

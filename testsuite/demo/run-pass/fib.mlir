module {

  // Class Fib
  // fib(x: U64 & imm): cown[U64Obj] & imm
  // FIXME: What is the semantics of the cown declaration above?
  func @"Fib::fib"(%x: !verona<"U64 & imm">) -> !verona<"Fib"> {

    // Avoid redeclaring constants (auto-generated probably would)
    %one = "verona.constant"() { "1" } : () -> (!verona<"U64 & imm">)
    %two = "verona.constant"() { "2" } : () -> (!verona<"U64 & imm">)

    // var pw = Promise::create();
    // Like "Fib::fib", we prefix all methods with the class mangled name
    %pw = "verona.Promise::create"() : () -> (!verona<"Promise">)

    // var pr = (mut-view pw).wait_handle();
    // FIXME: What is the semantics of this?
    %mv_pr = "verona.mut-view" (%pw) : (!verona<"Promise">) -> (!verona<"mut-view?">)
    %mv_pr_wh = "verona.wait_handle"() : () -> (!verona<"wait_handle?">)

    // if (x < 2)
    // Comparing verona types can't be standard "cmp", but the result _must_ be i1
    %eq = "verona.cmp_lt"(%x, %two) : (!verona<"U64 & imm">, !verona<"U64 & imm">) -> (i1)
    cond_br %eq, ^bb1, ^bb2

  ^bb1:                                 // pred: ^bb0
    // pw.fulfill(U64Obj.create(1));
    // object method call represented as class method with "this" as first argument
    %obj = "verona.U64Obj::create"(%one) : (!verona<"U64 & imm">) -> (!verona<"[U64Obj] & imm">)
    "verona.Promise::fullfill"(%pw, %obj) : (!verona<"Promise">, !verona<"[U64Obj] & imm">) -> ()
    br ^bb3

  ^bb2:                                 // pred: ^bb0:
    // when ()
    // Arguments on first basic block
    "when"() ({
      //   var p1 = Fib.fib(x - 1);
      // Same argument for "cmp" above applies to "sub" and "add" below
      %x1 = "verona.sub"(%x, %one) : (!verona<"U64 & imm">, !verona<"U64 & imm">) -> (!verona<"U64 & imm">)
      %p1 = call @"Fib::fib"(%x1) : (!verona<"U64 & imm">) -> (!verona<"Fib">)
     
      //   var p2 = Fib.fib(x - 2);
      %x2 = "verona.sub"(%x, %two) : (!verona<"U64 & imm">, !verona<"U64 & imm">) -> (!verona<"U64 & imm">)
      %p2 = call @"Fib::fib"(%x2) : (!verona<"U64 & imm">) -> (!verona<"Fib">)

      // when()
      // Arguments on first basic block
      "when"(%p1, %p2) ({
        // var r = U64Obj.create(p1.v + p2.v);
        // FIXME: what is the semantics of p1.v/p2.v?
        %p1v = "verona.getValue"(%p1) : (!verona<"Fib">) -> (!verona<"U64 & imm">)
        %p2v = "verona.getValue"(%p2) : (!verona<"Fib">) -> (!verona<"U64 & imm">)
        %sum = "verona.add"(%p1v, %p2v) : (!verona<"U64 & imm">, !verona<"U64 & imm">) -> (!verona<"U64 & imm">)
        %r = "verona.U64Obj::create"(%sum) : (!verona<"U64 & imm">) -> (!verona<"[U64Obj] & imm">)

        // pw.fulfill(r)
        // object method call represented as class method with "this" as first argument
        "verona.Promise::fullfill"(%pw, %r) : (!verona<"Promise">, !verona<"[U64Obj] & imm">) -> ()
      }) : (!verona<"Fib">, !verona<"Fib">) -> ()
    }) : () -> ()
    br ^bb3

  ^bb3:                                 // pred: ^bb1, ^bb2:
    // pr
    // FIXME: "pr" what now?
    %ret = "verona.pr"() : () -> (!verona<"Fib">)
    return %ret : !verona<"Fib">
  }

  // Class Main
  // main()
  func @"Main::main"() {
    // when (var uo = Fib.fib(12))
    %twelve = "verona.constant"() { "12" } : () -> (!verona<"U64 & imm">)
    %uo = call @"Fib::fib"(%twelve) : (!verona<"U64 & imm">) -> (!verona<"Fib">)
    "when"(%uo) ({
      // Builtin.print1("result={}\n", uo.v);
      // FIXME: what is the semantics of uo.v?
      %fmt = "verona.constant"() { "result={}\n" } : () -> (!verona<"string">)
      %uo_v = "verona.getValue"(%uo) : (!verona<"Fib">) -> (!verona<"U64">)
      "verona.Builtin::printl"(%fmt, %uo_v) : (!verona<"string">, !verona<"U64">) -> ()
    }) : (!verona<"Fib">) -> ()
  }
}

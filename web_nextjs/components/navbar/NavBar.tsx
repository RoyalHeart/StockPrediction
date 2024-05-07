import { NavActions } from "./NavActions";
import { User } from "@supabase/supabase-js";

export const NavBar = () => {
  return (
    <div className="sticky top-0 bg-black/30 backdrop-blur-sm">
      <header id="nav-bar-outer" className="mx-20">
        <div
          id="nav-bar-inner"
          className="grid-cols-nav font-normal h-fit grid"
        >
          <div
            id="brand"
            className="text-2xl/loose mr-5 bg-gradient-to-r from-green-400 from-25% to-purple-500 to-80% block text-transparent bg-clip-text"
          >
            <a href="/">
              <h1 className="">Trading AI</h1>
            </a>
          </div>
          <nav
            id="nav-pages"
            className="items-center flex self-center justify-start mx-10"
          >
            <ul className="inline-flex gap-5">
              <li>
                <a href="/about">ABOUT</a>
              </li>
              <li>
                <a href="/news">NEWS</a>
              </li>
            </ul>
          </nav>
          <NavActions></NavActions>
        </div>
      </header>
    </div>
  );
};

import React from "react";
import { Link } from "react-router-dom";
import { Home, Building2, User, Search, Menu } from "lucide-react";

export function Navigation() {
  return (
    <nav className="bg-white shadow-md fixed w-full z-50">
      <div className="max-w-7xl mx-auto px-6 flex justify-between items-center h-16">
        {/* 🔹 Logo */}
        <Link to="/" className="flex items-center">
          <Building2 className="h-8 w-8 text-blue-600" />
          <span className="ml-2 text-2xl font-bold text-gray-900">Homeverse AI</span>
        </Link>

        {/* 🔹 Desktop Menu */}
        <div className="hidden md:flex items-center space-x-6">
          <Link to="/" className="text-gray-700 hover:text-blue-600 font-medium">
            Home
          </Link>
          <Link to="/search" className="text-gray-700 hover:text-blue-600 font-medium">
            Search Properties
          </Link>
          <Link to="/dashboard" className="text-gray-700 hover:text-blue-600 font-medium">
            Dashboard
          </Link>
          <Link to="/profile" className="text-gray-700 hover:text-blue-600 font-medium flex items-center">
            <User className="h-5 w-5 mr-1" />
            Profile
          </Link>
        </div>

        {/* 🔹 Mobile Menu */}
        <div className="md:hidden">
          <Menu className="h-6 w-6 text-gray-700" />
        </div>
      </div>
    </nav>
  );
}

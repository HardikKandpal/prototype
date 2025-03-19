import React, { useEffect, useState } from "react";
import { propertyApi } from "../api/propertyApi";

export function AdminDashboard() {
  const [enquiries, setEnquiries] = useState([]);
  const [properties, setProperties] = useState([]);
  const [newProperty, setNewProperty] = useState({
    title: "",
    location: "",
    price: "",
    total_area: "",
  });

  useEffect(() => {
    async function fetchData() {
      setEnquiries(await propertyApi.getAdminEnquiries());
      setProperties(await propertyApi.getAdminProperties());
    }
    fetchData();
  }, []);

  const handleDeleteEnquiry = async (id: number) => {
    await propertyApi.deleteEnquiry(id);
    setEnquiries(enquiries.filter((e) => e.id !== id));
  };

  const handleAddProperty = async (e: React.FormEvent) => {
    e.preventDefault();
    await propertyApi.addProperty(newProperty);
    setProperties([...properties, newProperty]);
    setNewProperty({ title: "", location: "", price: "", total_area: "" });
  };

  return (
    <div className="max-w-6xl mx-auto py-10">
      <h2 className="text-3xl font-bold">Admin Dashboard</h2>

      <h3 className="text-xl font-semibold mt-6">User Enquiries</h3>
      <ul>
        {enquiries.map((e) => (
          <li key={e.id} className="border p-4 mt-2">
            <p><strong>{e.name}</strong> - {e.email}</p>
            <p>{e.message}</p>
            <button
              className="bg-red-500 text-white px-4 py-2 rounded"
              onClick={() => handleDeleteEnquiry(e.id)}
            >
              Delete
            </button>
          </li>
        ))}
      </ul>

      <h3 className="text-xl font-semibold mt-6">Properties</h3>
      <ul>
        {properties.map((p) => (
          <li key={p.id} className="border p-4 mt-2">
            <p><strong>{p.title}</strong> - {p.location}</p>
            <p>₹{p.price} | {p.total_area} sq ft</p>
          </li>
        ))}
      </ul>

      <h3 className="text-xl font-semibold mt-6">Add Property</h3>
      <form onSubmit={handleAddProperty} className="grid grid-cols-1 gap-4">
        <input
          type="text"
          placeholder="Title"
          value={newProperty.title}
          onChange={(e) => setNewProperty({ ...newProperty, title: e.target.value })}
          className="px-4 py-2 border rounded-lg"
        />
        <input
          type="text"
          placeholder="Location"
          value={newProperty.location}
          onChange={(e) => setNewProperty({ ...newProperty, location: e.target.value })}
          className="px-4 py-2 border rounded-lg"
        />
        <input
          type="number"
          placeholder="Price"
          value={newProperty.price}
          onChange={(e) => setNewProperty({ ...newProperty, price: e.target.value })}
          className="px-4 py-2 border rounded-lg"
        />
        <input
          type="number"
          placeholder="Total Area (sq ft)"
          value={newProperty.total_area}
          onChange={(e) => setNewProperty({ ...newProperty, total_area: e.target.value })}
          className="px-4 py-2 border rounded-lg"
        />
        <button className="bg-green-600 text-white px-4 py-2 rounded">
          Add Property
        </button>
      </form>
    </div>
  );
}
